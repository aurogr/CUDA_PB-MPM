#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "solver.cuh"
#include "boundary.h"

#pragma region Kernel Functions
__device__ float2 ApplyNodeCollision(
    const float2& Xi,
    float2 Vi_col,
    const BoundaryData& bounds,
    const float dt)
{
    for (int b = 0; b < bounds.count; ++b)
    {
        LineSegmentBoundary border = bounds.d_borders[b];

        // Vector from first corner/start point to node: (Xi - X_corner[0])
        float2 rel_pos = make_float2(Xi.x - border.start.x, Xi.y - border.start.y);

        // Current distance between node and boundary: normal.dot(Xi - X_corner[0])
        float distance = border.normal.x * rel_pos.x + border.normal.y * rel_pos.y;

        // Type 1: Sticky Boundary
        if (border.type == STICKY && distance < 0.0f)
        {
            Vi_col = make_float2(0.0f, 0.0f);
        }
        // Types 2 & 3: Separating / Sliding Boundary
        else
        {
            // Compute trial distance: trial_position = Xi + DT * Vi_col
            float2 trial_position = make_float2(Xi.x + dt * Vi_col.x, Xi.y + dt * Vi_col.y);
            float2 trial_rel_pos = make_float2(trial_position.x - border.start.x, trial_position.y - border.start.y);

            float trial_distance = border.normal.x * trial_rel_pos.x + border.normal.y * trial_rel_pos.y;
            float dist_c = trial_distance - fminf(distance, 0.0f);

            // Record collision & update velocity
            if ((border.type == SEPARATING && dist_c < 0.0f) ||
                (border.type == SLIDING && distance < 0.0f))
            {
                Vi_col.x -= (dist_c / dt) * border.normal.x;
                Vi_col.y -= (dist_c / dt) * border.normal.y;
            }
        }
    }

    return Vi_col;
}

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
    const float RHO_water = 1.0;					// Density
    const float K_water = 50.0;					// Bulk Modulus
    const int   GAMMA_water = 3;					// Penalize deviation form incompressibility

    float dJp = -K_water * (1.0 / pow(Jp, GAMMA_water) - 1.0);	// Deformation gradient increment
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

            // TODO: see if this step is necessary or i could skip it
            if (Wip < 1e-5f) continue; // skip negligible contributions

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

__global__ void updateGrid_kernel(const float* d_Mi, float2* d_Vi, float2* d_Vi_Col, float2* d_Vi_Fri, float2* d_Fi, const float dt, const float G, const int num_nodes, const int gridX, const int gridY, BoundaryData bounds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    float Mi = d_Mi[i];

    if (Mi < 1e-7f) {
        d_Vi[i] = make_float2(0.0f, 0.0f);
        d_Vi_Col[i] = make_float2(0.0f, 0.0f);
        d_Vi_Fri[i] = make_float2(0.0f, 0.0f);
        return;
    }

    // 1. Get nodal velocity from momentum: v_i = (mv)_i / m_i
    d_Vi[i].x /= Mi;
    d_Vi[i].y /= Mi;

    // 2. Update velocity
    // time integration on grid: 
    // v_i^(n+1) = v_ì^(n) + (dt/mi)*(f_i + f_ext)
    // f_ext = m_i * G

    float timeStepByMass = dt / Mi;
    d_Fi[i].x = timeStepByMass * (-d_Fi[i].x);
    d_Fi[i].y = timeStepByMass * -d_Fi[i].y + dt * G;

    d_Vi[i].x += d_Fi[i].x;
    d_Vi[i].y += d_Fi[i].y;

    // 3. Compute collision velocity
    // 3.1 Get node coordinates
    int gx = i % (gridX + 1);
    int gy = i / (gridX + 1);
    float2 Xi = make_float2((float)gx, (float)gy);

    // 3.2 Compute and update
    float2 Vi_col = ApplyNodeCollision(Xi, d_Vi[i], bounds, dt);
    d_Vi_Col[i] = Vi_col;

    // 4. TODO: Apply friction
    d_Vi_Fri[i] = Vi_col;
}

__global__ void g2p_kernel(float2* d_Xp, float2* d_Vp, float* d_Jp, float4* d_Bp, // Matrix 2x2 stored as float4: x=00, y=01, z=10, w=11
    float2* d_Vi_col, float2* d_Vi_fri, const int num_particles, const int gridX, const int gridY, const float dt)
{
    // 1. Get particle (thread per particle) and its characteristics
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= num_particles) return;

    float2 Xp = d_Xp[p];
    float2 new_Xp = make_float2(0.0f, 0.0f);
    float2 new_Vp_fri = make_float2(0.0f, 0.0f);
    float2 new_Vp_col = make_float2(0.0f, 0.0f);
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
            new_Vp_fri.x += Wip * d_Vi_fri[node_idx].x;
            new_Vp_fri.y += Wip * d_Vi_fri[node_idx].y;

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

            // 3.7 Accumulate velocity for each particle based on the weight of the current node
            new_Vp_col.x += Wip * d_Vi_col[node_idx].x;
            new_Vp_col.y += Wip * d_Vi_col[node_idx].y;
        }
    }

    // 4. Write back the value calculated for the particle based on the nodes that influence it
    d_Vp[p] = new_Vp_fri;
    d_Bp[p] = new_Bp;

    // 5. Position advection using collision velocity
    Xp.x += dt * new_Vp_col.x;
    Xp.y += dt * new_Vp_col.y;

    // 6. Clamp particle position inside the active domain boundary
    const float padding = 2.0f;
    Xp.x = fminf(fmaxf(Xp.x, padding), (float)gridX - padding);
    Xp.y = fminf(fmaxf(Xp.y, padding), (float)gridY - padding);

    d_Xp[p] = Xp;

    // 7. Nodal deformation
    float next_Jp = d_Jp[p] * (1.0f + dt * (T.x + T.w));

    // 8. Prevent Volume Explosions
    d_Jp[p] = fminf(fmaxf(next_Jp, 0.6f), 1.5f);
}

#pragma endregion

#pragma region Host Solver Implementation

void p2g(const ParticleSystem& ps, Grid& grid, float dt)
{
    if (ps.num_particles == 0) return;
    int blockSize = 256;
    int gridSize = (ps.num_particles + blockSize - 1) / blockSize;

    p2g_kernel << <gridSize, blockSize >> > (
        ps.d_Vp0, ps.d_Xp, ps.d_Vp, ps.d_Mp, ps.d_Bp,
        ps.Jp, ps.Ap, grid.d_Mi, grid.d_Vi, grid.d_Fi,
        dt, ps.num_particles, grid.grid_x, grid.grid_y
        );
}

void updateGrid(Grid& grid, float dt, BoundaryData bounds)
{
    int blockSize = 256;
    int gridSize = (grid.num_nodes + blockSize - 1) / blockSize;

    updateGrid_kernel << <gridSize, blockSize >> > (
        grid.d_Mi, grid.d_Vi, grid.d_Vi_col, grid.d_Vi_fri, grid.d_Fi,
        dt, -9.81f, grid.num_nodes, grid.grid_x, grid.grid_y, bounds
        );
}

void g2p(ParticleSystem& ps, const Grid& grid, float dt)
{
    if (ps.num_particles == 0) return;
    int blockSize = 256;
    int gridSize = (ps.num_particles + blockSize - 1) / blockSize;

    g2p_kernel << <gridSize, blockSize >> > (
        ps.d_Xp, ps.d_Vp, ps.Jp, ps.d_Bp,
        grid.d_Vi_col, grid.d_Vi_fri, ps.num_particles, grid.grid_x, grid.grid_y, dt
        );
}

#pragma endregion