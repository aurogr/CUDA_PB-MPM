#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "solver.cuh"
#include "boundary.h"
#include "types.h"

__device__ float ComputeAp(const WaterData& mat, int p, float Vp0) {

    float Jp = mat.d_Jp[p];

    float pressure = -mat.K * (1.0f / pow(Jp, mat.GAMMA) - 1.0f);	// Tait's pressure formula

    float currentVolume = Vp0 * Jp;

    float Ap = pressure * currentVolume;

    return Ap;
}

__device__ Matrix2f ComputeAp(const SnowData& mat, int p, float Vp0) {
    float Jp = mat.d_Jp[p];
    Matrix2f Fe = mat.d_Fe[p];
    Matrix2f Fp;

    // Compute Lame parameters
    float plastic_hardening_factor = std::exp(mat.KSI * (1.0f - Jp));
    float mu = mat.MU_0 * plastic_hardening_factor;
    float lambda = mat.LAMBDA_0 * plastic_hardening_factor;

    // Polar decomposition to extract rotation (Fe = Re * Se)
    Matrix2f Re, Se;
    Fe.polar_decomp(&Re, &Se);

    // Compute kirchhoff stress tensor
    float Je = Fe.det();
    auto stress_tensor = 2.0f * mu * (Fe - Re) * Fe.transpose() + lambda * Je * (Je - 1.0f) * identity();

    // Compute Ap
    Matrix2f Ap = Vp0 * stress_tensor;

    return Ap;
}

__device__ void UpdateDeformation(const WaterData& mat, int p, float dt, Matrix2f T, Matrix2f* d_Bp) {
    mat.d_Jp[p] *= (1.0f + dt * (T.m00 + T.m11));
}

__device__ void UpdateDeformation(const SnowData& mat, int p, float dt, Matrix2f T, Matrix2f* d_Bp) {
    Matrix2f FeTr = (identity() + dt * T) * mat.d_Fe[p];
    Matrix2f FpTr = mat.d_Fp[p];

    Matrix2f U, V;
    Vector2f Eps;
    FeTr.svd(&U, &Eps, &V);

    Vector2f proj = Eps.clamp(1.0f - mat.THT_C, 1.0f + mat.THT_S);

    Matrix2f Fe = U.diag_product(proj) * V.transpose();

    Vector2f plastic_factor(
        proj.x / (Eps.x + 1e-12f),
        proj.y / (Eps.y + 1e-12f)
    );
    Matrix2f DiagInvRatio(plastic_factor.x, 0.0f, 0.0f, plastic_factor.y);
    
    Matrix2f Fp = V * DiagInvRatio * V.transpose() * FpTr;

    // Write back to GPU
    mat.d_Fe[p] = Fe;
    mat.d_Fp[p] = Fp;
    mat.d_Jp[p] = Fp.det();
}


#pragma region Kernel Functions
__device__ void ApplyBoundaryCollisions(
    const Vector2f& Xi, const Vector2f& Vi_in, const BoundaryData& bounds, const float dt,
    Vector2f& Vi_col, Vector2f& Vi_fri)
{
    Vi_col = Vi_in;
    Vi_fri = Vi_in;

    for (int b = 0; b < bounds.count; ++b)
    {
        LineSegmentBoundary border = bounds.d_borders[b];
        float distance = border.normal.dot(Xi - border.start);

        float trial_dist = border.normal.dot(Xi + dt * Vi_col - border.start);
        float dist_c = trial_dist - fminf(distance, 0.0f);

        // Trigger collision only if the node is touching or penetrating the boundary
        if (distance < 0.0f || dist_c < 0.0f)
        {
            // Standard collision response (pushes out of the wall)
            Vi_col -= (dist_c / dt) * border.normal;

            // Friction response based on collision velocity
            Vector2f v_col = Vi_fri - (dist_c / dt) * border.normal;
            float normal_vel = v_col.dot(border.normal);

            // We only care about friction if the node is pushing *into* the wall (normal_vel < 0)
            // or sliding along it.
            Vector2f tangent = v_col - normal_vel * border.normal;
            float t_speed = tangent.length();

            if (t_speed > 1e-6f) {
                // Normalize tangent to get the exact sliding direction unit vector
                Vector2f tangent_dir = tangent / t_speed;

                // Coulomb friction limit: max friction force scales with normal force magnitude
                float max_friction_force = border.friction * fabsf(normal_vel);

                // Reduce tangential speed by the friction drop, clamped to 0 so it doesn't reverse direction
                float new_t_speed = fmaxf(0.0f, t_speed - max_friction_force);

                // Reconstruct final friction velocity
                Vi_fri = normal_vel * border.normal + new_t_speed * tangent_dir;
            }
            else {
                Vi_fri = v_col;
            }
        }
    }
}


template <typename MatData>
__global__ void p2g_kernel(const float* d_Vp0, const Vector2f* d_Xp, const Vector2f* d_Vp, const float* d_Mp, const Matrix2f* d_Bp, MatData d_mat,
    float* d_Mi, Vector2f* d_Vi, Vector2f* d_Fi, const float dt, const int num_particles, const int gridX, const int gridY)
{
    // 1. Get particle (thread per particle) and its characteristics
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= num_particles) return;

    float Vp0 = d_Vp0[p];
    Vector2f Xp = d_Xp[p];
    Vector2f Vp = d_Vp[p];
    Matrix2f Bp = d_Bp[p];
    float Mp = d_Mp[p];

    auto Ap = ComputeAp(d_mat, p, Vp0);

    // 2. Find the bottom-left node closest to the particle of the 3x3 stencil (and init weights)
    Vector2f w[3], dw[3];
    Vector2f base = InitQuadraticWeights(Xp, w, dw);

    // 3. Loop over the neighbor nodes
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {

            // 3.1 Get current node idx
            Vector2f node(base.x + x, base.y + y);

            // Out-of-bounds boundary check
            if (node.x < 0 || node.x > gridX || node.y < 0 || node.y > gridY) continue;

            int node_idx = node.x + (gridX + 1) * node.y;

            // 3.2 Get weights and derivatives
            float Wip = w[x].x * w[y].y;
            Vector2f dWip(dw[x].x * w[y].y, w[x].x * dw[y].y);

            // APIC Matrix-Vector product: Bp * (x_i - x_p)
            Vector2f Bp_dist = Bp * (node - Xp);
            
            // Mass accumulation: mi = sum(Wip * Mp)
            float inMi = Wip * Mp;

            // Velocity (Momentum) accumulation: mi * vi = sum(Wip * Mp * (Vp + Bp * Dp^-1 * (xi - xp)))
            // Dp^-1 is 4.0/h^2 and h = 1.0
            Vector2f inVi = inMi * (Vp + 4.0f * Bp_dist);

            // Force/Pressure contribution accumulation
            Vector2f inFi = Ap * dWip;

            // Atomic accumulation into GPU grid nodes
            atomicAdd(&d_Mi[node_idx], inMi);

            atomicAdd(&d_Vi[node_idx].x, inVi.x);
            atomicAdd(&d_Vi[node_idx].y, inVi.y);

            atomicAdd(&d_Fi[node_idx].x, inFi.x);
            atomicAdd(&d_Fi[node_idx].y, inFi.y);
        }
    }
}

__global__ void updateGrid_kernel(const float* d_Mi, Vector2f* d_Vi, Vector2f* d_Vi_Col, Vector2f* d_Vi_Fri, Vector2f* d_Fi,
    const float dt, const float G, const int num_nodes, const int gridX, const int gridY, BoundaryData bounds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    float Mi = d_Mi[i];

    if (Mi < 1e-10f) {
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

    d_Vi[i] += d_Fi[i];

    // 3. Compute collision velocity
    // 3.1 Get node coordinates
    int gx = i % (gridX + 1);
    int gy = i / (gridX + 1);
    Vector2f Xi ((float)gx, (float)gy);

    // 3.2 Compute and update
    Vector2f Vi_col, Vi_fri;
    ApplyBoundaryCollisions(Xi, d_Vi[i], bounds, dt, Vi_col, Vi_fri);
    d_Vi_Col[i] = Vi_col;
    d_Vi_Fri[i] = Vi_fri;
}

template <typename MatData>
__global__ void g2p_kernel(Vector2f* d_Xp, Vector2f* d_Vp, Matrix2f* d_Bp, MatData d_mat,
    Vector2f* d_Vi_col, Vector2f* d_Vi_fri, const int num_particles, const int gridX, const int gridY, const float dt)
{
    // 1. Get particle (thread per particle) and its characteristics
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= num_particles) return;

    Vector2f Xp = d_Xp[p];
    Vector2f new_Xp (0.0f, 0.0f);
    Vector2f new_Vp_fri (0.0f, 0.0f);
    Vector2f new_Vp_col (0.0f, 0.0f);
    Matrix2f new_Bp (0.0f, 0.0f, 0.0f, 0.0f);

    Matrix2f T (0.0f, 0.0f, 0.0f, 0.0f); // nodal deformation

    // 2. Get base grid node and init weights
    Vector2f w[3], dw[3];
    Vector2f base = InitQuadraticWeights(Xp, w, dw);

    // 3. Stencil
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {
            // 3.1 Get current node idx
            Vector2f node(base.x + x, base.y + y);

            // Out-of-bounds boundary check
            if (node.x < 0 || node.x > gridX || node.y < 0 || node.y > gridY) continue;

            int node_idx = node.x + (gridX + 1) * node.y;

            // 3.2 Get weights
            float Wip = w[x].x * w[y].y;

            // TODO: see if this step is necessary or i could skip it
            if (Wip < 1e-10f) continue; // skip negligible contributions

            // 3.3 Accumulate particle velocity: Vp += Wip * Vi
            new_Vp_fri += Wip * d_Vi_fri[node_idx];

            // 3.4 Accumulate B_p matrix: B_p += W_ip * (V_i outer_product dist)
            new_Bp += Wip * outer_product(d_Vi_fri[node_idx], node - Xp);

            // 3.5 Calculate 2D gradient components of the weight function
            Vector2f gradW(
                dw[x].x * w[y].y,  // dWx/dx * Wy
                w[x].x * dw[y].y   // Wx * dWy/dy
            );

            // 3.6 Accumulate velocity gradient matrix T = Vi ⊗ ∇Wip
            T += Matrix2f::outer_product(d_Vi_col[node_idx], gradW);

            // 3.7 Accumulate velocity for each particle based on the weight of the current node
            new_Vp_col += Wip * d_Vi_col[node_idx];
        }
    }

    // 4. Write back the value calculated for the particle based on the nodes that influence it
    d_Vp[p] = new_Vp_fri;
    d_Bp[p] = new_Bp;

    // 5. Position advection using collision velocity
    Xp += dt * new_Vp_col;

    // 6. Clamp particle position inside the active domain boundary
    const float padding = 2.0f;
    Xp.x = fminf(fmaxf(Xp.x, padding), (float)gridX - padding);
    Xp.y = fminf(fmaxf(Xp.y, padding), (float)gridY - padding);

    d_Xp[p] = Xp;

    // 7. Update deformation
    UpdateDeformation(d_mat, p, dt, T, &d_Bp[p]);
}

#pragma endregion

#pragma region Host Solver Implementation

template <typename MatData>
void p2g(const ParticleSystem<MatData>& ps, Grid& grid, float dt)
{
    if (ps.num_particles == 0) return;
    int blockSize = 256;
    int gridSize = (ps.num_particles + blockSize - 1) / blockSize;

    p2g_kernel << <gridSize, blockSize >> > (
        ps.d_Vp0, ps.d_Xp, ps.d_Vp, ps.d_Mp, ps.d_Bp, ps.d_Mat,
        grid.d_Mi, grid.d_Vi, grid.d_Fi, dt, ps.num_particles, grid.grid_x, grid.grid_y
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

template <typename MatData>
void g2p(ParticleSystem<MatData>& ps, const Grid& grid, float dt)
{
    if (ps.num_particles == 0) return;
    int blockSize = 256;
    int gridSize = (ps.num_particles + blockSize - 1) / blockSize;

    g2p_kernel << <gridSize, blockSize >> > (
        ps.d_Xp, ps.d_Vp, ps.d_Bp, ps.d_Mat,
        grid.d_Vi_col, grid.d_Vi_fri, ps.num_particles, grid.grid_x, grid.grid_y, dt
        );
}

#pragma endregion

#pragma region Explicit instantiation
template void p2g<WaterData>(const ParticleSystem<WaterData>& ps, Grid& grid, float dt);
template void p2g<SnowData>(const ParticleSystem<SnowData>& ps, Grid& grid, float dt);

template void g2p<WaterData>(ParticleSystem<WaterData>& ps, const Grid& grid, float dt);
template void g2p<SnowData>(ParticleSystem<SnowData>& ps, const Grid& grid, float dt);

#pragma endregion