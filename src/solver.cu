#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "solver.cuh"

__global__ void p2g_kernel(const float* d_Vp0, const float2* d_Xp, const float2* d_Vp, const float* d_Mp, const float4* d_Bp, // Matrix 2x2 stored as float4: x=00, y=01, z=10, w=11
    float* d_Jp, float* d_Ap, // these are inherent to the constitutive model and should not be here in a future
    float* d_Mi, float2* d_Vi, float2* d_Fi, const float dt, const int num_particles, const int gridX, const int gridY)
{
    // 1. Get particle (thread per particle) and its characteristics
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= num_particles) return;

    float Vp0 = d_Vp0[p];
    float2 Xp = d_Xp[p];
    float2 Vp = d_Vp[p];
    float4 Bp = d_Bp[p];
    float Mp = d_Mp[p];
    float Ap = d_Ap[p];
    float Jp = d_Jp[p];

    // 1.2. Calculate stress derivative from particle constitutive model
    // (IDEA: Here each particle could have an index that points to an enum of constitutive types? or a pointer to a type?)
    // For now we only have water
    const double RHO_water = 1.0;					// Density
    const double K_water = 50.0;					// Bulk Modulus
    const int   GAMMA_water = 3;					// Penalize deviation form incompressibility

    double dJp = -K_water * (1.0 / pow(Jp, GAMMA_water) - 1.0);	// Deformation gradient increment
    Ap = dJp * Vp0 * Jp;

    // 2. Find the bottom-left node closest to the particle of the 3x3 stencil (and init weights)
    float w_x[3], w_y[3], dw_x[3], dw_y[3];
    int base_x = InitQuadraticWeights(Xp.x, w_x, dw_x);
    int base_y = InitQuadraticWeights(Xp.y, w_y, dw_y);

    // 3. Loop over the neighbor nodes
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {

            // 3.1 Get current node idx
            int node_x = base_x + x;
            int node_y = base_y + y;

            // Out-of-bounds boundary check
            if (node_x < 0 || node_x > gridX || node_y < 0 || node_y > gridY) continue;

            int node_idx = node_x + (gridX + 1) * node_y;

            // 3.2 Get weights
            float Wip = w_x[x] * w_y[y];
            float dWip_x = dw_x[x] * w_y[y];
            float dWip_y = w_x[x] * dw_y[y];

            // Distance vector from particle to node: (x_i - x_p)
            float xi_minus_xp_x = ((float)node_x) - Xp.x;
            float xi_minus_xp_y = ((float)node_y) - Xp.y;

            // APIC Matrix-Vector product: Bp * (x_i - x_p)
            float Bp_dist_x = Bp.x * xi_minus_xp_x + Bp.y * xi_minus_xp_y;
            float Bp_dist_y = Bp.z * xi_minus_xp_x + Bp.w * xi_minus_xp_y;
            
            // Mass accumulation: mi = sum(Wip * Mp)
            float inMi = Wip * Mp;

            // Velocity (Momentum) accumulation: mi * vi = sum(Wip * Mp * (Vp + Bp * Dp^-1 * (xi - xp)))
            // Dp^-1 is 4.0/h^2 and h = 1.0
            float inVi_x = inMi * (Vp.x + 4.0f * Bp_dist_x);
            float inVi_y = inMi * (Vp.y + 4.0f * Bp_dist_y);

            // Force/Pressure contribution accumulation
            float inFi_x = Ap * dWip_x;
            float inFi_y = Ap * dWip_y;

            // Atomic accumulation into GPU grid nodes
            atomicAdd(&d_Mi[node_idx], inMi);

            atomicAdd(&d_Vi[node_idx].x, inVi_x);
            atomicAdd(&d_Vi[node_idx].y, inVi_y);

            atomicAdd(&d_Fi[node_idx].x, inFi_x);
            atomicAdd(&d_Fi[node_idx].y, inFi_y);
        }
    }
}

__global__ void updateGrid_kernel(const float* d_Mi, float2* d_Vi, const float2* d_Fi, const float dt, const float G, const int num_nodes, const int gridX, const int gridY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    float Mi = d_Mi[i];

    if (Mi <= 0) return;

    // 1. Get nodal velocity from momentum: v_i = (mv)_i / m_i
    float Vi_x = d_Vi[i].x / Mi;
    float Vi_y = d_Vi[i].y / Mi;

    // 2. Perform time integration on grid: 
    // v_i^(n+1) = v_ì^(n) + (dt/mi)*(f_i + f_ext)
    // f_ext = m_i * G

    float timeStepByMass = dt / Mi;
    Vi_x += timeStepByMass * (d_Fi[i].x);
    Vi_y += timeStepByMass * (d_Fi[i].y / Mi + G) + dt * G;

    d_Vi[i].x = Vi_x;
    d_Vi[i].y = Vi_y;

    // TODO: Apply collisions and frictions
}


__global__ void g2p_kernel(float2* d_Xp, float2* d_Vp, float* d_Jp, float4* d_Bp, // Matrix 2x2 stored as float4: x=00, y=01, z=10, w=11
    float2* d_Vi_fri, float2* d_Vi_col, const int num_particles, const int gridX, const int gridY, const float dt)
{
    // 1. Get particle (thread per particle) and its characteristics
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= num_particles) return;

    float2 Xp = d_Xp[p];
    float2 new_Xp = make_float2(0.0f, 0.0f);
    float2 new_Vp = make_float2(0.0f, 0.0f);
    float4 new_Bp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float4 T = make_float4(0.0f, 0.0f, 0.0f, 0.0f); // nodal deformation

    // 2. Get base grid node and init weights
    float w_x[3], w_y[3], dw_x[3], dw_y[3];
    int base_x = InitQuadraticWeights(Xp.x, w_x, dw_x);
    int base_y = InitQuadraticWeights(Xp.y, w_y, dw_y);

    // 3. Stencil
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {
            // 3.1 Get current node idx
            int node_x = base_x + x;
            int node_y = base_y + y;

            // Out-of-bounds boundary check
            if (node_x < 0 || node_x > gridX || node_y < 0 || node_y > gridY) continue;

            int node_idx = node_x + (gridX + 1) * node_y;

            // 3.2 Get weights
            float Wip = w_x[x] * w_y[y];

            // 3.3 Accumulate particle velocity: Vp += Wip * Vi
            new_Vp.x += Wip * (d_Vi_fri[node_idx].x);
            new_Vp.y += Wip * d_Vi_fri[node_idx].y;

            // 3.4 Accumulate B_p matrix: B_p += W_ip * (V_i outer_product dist)
            float dist_x = (float)node_x - Xp.x;
            float dist_y = (float)node_y - Xp.y;

            float term_x = Wip * d_Vi_fri[node_idx].x;
            float term_y = Wip * d_Vi_fri[node_idx].y;

            new_Bp.x += term_x * dist_x; // Bp[0][0]
            new_Bp.y += term_x * dist_y; // Bp[0][1]
            new_Bp.z += term_y * dist_x; // Bp[1][0]
            new_Bp.w += term_y * dist_y; // Bp[1][1]

            // 3.5 Calculate 2D gradient components of the weight function
            float gradW_x = dw_x[x] * w_y[y];
            float gradW_y = w_x[x] * dw_y[y];

            // 3.6 Accumulate velocity gradient matrix T = Vi ⊗ ∇Wip
            T.x += d_Vi_col[node_idx].x * gradW_x; // T[0][0] = dvx/dx
            T.y += d_Vi_col[node_idx].x * gradW_y; // T[0][1] = dvx/dy
            T.z += d_Vi_col[node_idx].y * gradW_x; // T[1][0] = dvy/dx
            T.w += d_Vi_col[node_idx].y * gradW_y; // T[1][1] = dvy/dy

            // Particle advection:
            new_Xp.x += Wip * (node_x + dt * d_Vi_col[node_idx].x);
            new_Xp.y += Wip * (node_y + dt * d_Vi_col[node_idx].y);
        }
    }

    // 4. Write back the value calculated for the particle based on the nodes that influence it
    d_Vp[p] = new_Vp;
    d_Bp[p] = new_Bp;
    d_Xp[p] = new_Xp;

    // 5. Nodal deformation
    d_Jp[p] *= 1.0f + dt * (T.x + T.w);
}

void run_p2g(const ParticleSystem& ps, Grid& grid, float h) {
    // 1. Clear grid nodes to 0 before accumulating
    cudaMemset(grid.d_Mi, 0, grid.num_nodes * sizeof(float));
        cudaMemset(grid.d_Vi, 0, grid.num_nodes * sizeof(float2));
        cudaMemset(grid.d_Fi, 0, grid.num_nodes * sizeof(float2));

        // 2. Kernel launch configuration
        int threadsPerBlock = 256;
    int blocksPerGrid = (ps.num_particles + threadsPerBlock - 1) / threadsPerBlock;

    // 3. Launch P2G Kernel
    p2g_kernel << <blocksPerGrid, threadsPerBlock >> > (
        ps.d_Xp, ps.d_Vp, ps.d_Bp, ps.d_Mp,
        grid.d_Mi, grid.d_Vi, grid.d_Fi,
        ps.Ap, ps.Jp,
        ps.num_particles,
        grid.grid_x, grid.grid_y
        );

    // 4. Synchronize and check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "P2G Launch Error: " << cudaGetErrorString(err) << "\n";
    }
    cudaDeviceSynchronize();
}
