#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "solver.cuh"
#include "boundary.h"
#include "types.h"

__device__ float ComputeAp(const WaterData& mat, int p, float Vp0) {

    float Jp = mat.d_Jp[p];

    float pressure = -mat.K * (1.0 / pow(Jp, mat.GAMMA) - 1.0);	// Tait's pressure formula

    float currentVolume = Vp0 * Jp;

    float Ap = pressure * currentVolume;

    return Ap;
}

__device__ Matrix2f ComputeAp(const SnowData& mat, int p, float Vp0) {
    float Jp = mat.d_Jp[p];
    Matrix2f Fe = mat.d_Fe[p];
    Matrix2f Fp;

    // Compute Lame parameters
    float plastic_hardening_factor = std::exp(mat.KSI * (1.0 - Jp));
    float mu = mat.MU_0 * plastic_hardening_factor;
    float lambda = mat.LAMBDA_0 * plastic_hardening_factor;

    // Polar decomposition to extract rotation (Fe = Re * Se)
    Matrix2f Re, Se;
    Fe.polar_decomp(&Re, &Se);

    // Compute kirchhoff stress tensor
    float Je = Fe.det();
    auto stress_tensor = 2 * mu * (Fe - Re) * Fe.transpose() + lambda * Je * (Je - 1.0) * identity();

    // Compute Ap
    Matrix2f Ap = Vp0 * stress_tensor;

    return Ap;
}

__device__ void UpdateDeformation(const WaterData& mat, int p, float dt, Matrix2f T) {
    float next_Jp = mat.d_Jp[p] * (1.0f + dt * (T.m00 + T.m11));

    mat.d_Jp[p] = fminf(fmaxf(next_Jp, 0.6f), 1.5f); // prevent volume explosions
}

__device__ void UpdateDeformation(const SnowData& mat, int p, float dt, Matrix2f T) {
    // 1. Advance elastic deformation gradient
    Matrix2f Fe_trial = (identity() + dt * T) * mat.d_Fe[p];

    // 2. Perform SVD
    Matrix2f U, Sigma, V;
    Fe_trial.svd(&U, &Sigma, &V);

    // Ensure trial singular values are strictly positive
    float s0_trial = fabsf(Sigma.m00);
    float s1_trial = fabsf(Sigma.m11);

    // 3. Clamp principal stretches to yield surface
    float min_stretch = 1.0f - mat.THT_C;
    float max_stretch = 1.0f + mat.THT_S;

    float s0_clamped = fminf(fmaxf(s0_trial, min_stretch), max_stretch);
    float s1_clamped = fminf(fmaxf(s1_trial, min_stretch), max_stretch);

    // 4. Reconstruct clamped elastic deformation matrix
    Matrix2f Sigma_clamped(s0_clamped, 0.0f,
        0.0f, s1_clamped);

    Matrix2f Fe_new = U * Sigma_clamped * V.transpose();

    // 5. Update Plastic Determinant (J_P)
    // J_p^(n+1) = J_p^n * (det(F_e_trial) / det(F_e_clamped))
    float det_trial = s0_trial * s1_trial;
    float det_clamped = s0_clamped * s1_clamped;

    float Jp_new = mat.d_Jp[p];
    if (det_clamped > 1e-6f) {
        Jp_new *= (det_trial / det_clamped);
    }

    // Write back to GPU global memory
    mat.d_Fe[p] = Fe_new;
    mat.d_Jp[p] = Jp_new;
}


#pragma region Kernel Functions
__device__ Vector2f ApplyNodeCollision(
    const Vector2f& Xi,
    Vector2f Vi_col,
    const BoundaryData& bounds,
    const float dt)
{
    for (int b = 0; b < bounds.count; ++b)
    {
        LineSegmentBoundary border = bounds.d_borders[b];

        // Vector from first corner/start point to node:
        Vector2f rel_pos = Xi - border.start;

        // Current distance between node and boundary:
        float distance = border.normal.dot(rel_pos);

        // Type 1: Sticky Boundary
        if (border.type == STICKY && distance < 0.0f)
        {
            Vi_col = Vector2f(0.0f, 0.0f);
        }
        // Types 2 & 3: Separating / Sliding Boundary
        else
        {
            // Compute trial distance:
            Vector2f trial_position = Xi + dt * Vi_col;
            Vector2f trial_rel_pos = trial_position - border.start;

            float trial_distance = border.normal.dot(trial_rel_pos);
            float dist_c = trial_distance - fminf(distance, 0.0f);

            // Record collision & update velocity
            if ((border.type == SEPARATING && dist_c < 0.0f) ||
                (border.type == SLIDING && distance < 0.0f))
            {
                Vi_col -= (dist_c / dt) * border.normal;
            }
        }
    }

    return Vi_col;
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

    d_Vi[i] += d_Fi[i];

    // 3. Compute collision velocity
    // 3.1 Get node coordinates
    int gx = i % (gridX + 1);
    int gy = i / (gridX + 1);
    Vector2f Xi ((float)gx, (float)gy);

    // 3.2 Compute and update
    Vector2f Vi_col = ApplyNodeCollision(Xi, d_Vi[i], bounds, dt);
    d_Vi_Col[i] = Vi_col;

    // 4. TODO: Apply friction
    d_Vi_Fri[i] = Vi_col;
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
            if (Wip < 1e-7f) continue; // skip negligible contributions

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
    UpdateDeformation(d_mat, p, dt, T);
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