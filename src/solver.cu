#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "solver.cuh"
#include "boundary.h"
#include "types.h"


#pragma region Material

#pragma region Water
__device__ void computeDisplacementWater(const MaterialSettings& settings, const float Jp, Matrix2f& Dp) 
{
    //(1) Viscosity: remove deviatoric part of deformation displacement
    Matrix2f deviatoric = -1.0f * (Dp + Dp.transpose());
    Dp += settings.viscosity * 0.5f * deviatoric;
    
    //(2) Volume preservation (Incompressibility)
    
    //Liquids are incompressible, their volume ratio J (det(F)) needs to equal 1.0f
    //we need to calculate an impulse (alpha) that forces the liquid back to its resting volume
    
    //The two identities used for that are:
    //1. det(F_new) = 1.0f;
    //2. det(F_new) = det(I + D_p) * det(F_old); where we can linearize the determinant as: det(I + D_p) = 1.0f + Tr(D_p) 
    
    float alpha = 0.5f * (1.0f / Jp - Dp.trace() - 1.0f); // where the 0.5f comes from using an identity matrix in 2d space so the trace increases by 2*alpha

    // Finally, we add to the deformation displacement towards preserving the volume
    Dp += settings.relaxation * alpha * identity();
}

__device__ void updateDeformationWater(const MaterialSettings& settings, float& Jp, const Matrix2f Dp) {
    // Liquids hold no memory of shape, they only care about volume, so storing only the determinant of the deformation gradient is enough

    // The true update for volume is det(F_new) = det(I + Dp) * det(F_old)
    // but as before, we can use the trace instead of the complicated determinant
    Jp *= (1.0f + Dp.trace());
    Jp = fmaxf(Jp, 0.1f); // Never allow J <= 0
}
#pragma endregion

#pragma region Snow
__device__ void computeDisplacementSnow(const MaterialSettings& settings, const Matrix2f Fp, const Matrix2f Fe, Matrix2f& Dp)
{
    // 1. Trial deformation gradient
    Matrix2f F_trial = (identity() + Dp) * Fe;

    // 2. Corotated elastic target
    Matrix2f U, V;
    Vector2f Sigma;
    F_trial.svd(&U, &Sigma, &V);
    Matrix2f A = U * V.transpose();

    // 3. Elastric strain
    Matrix2f target_D = A * Fe.inverse() - identity();
    Matrix2f diff = target_D - Dp;

    // 4. Disney hardening
    float Jp = fmaxf(Fp.det(), 0.01f);
    float hardening = expf(settings.hard_coeff * (1.0f - Jp));

    // 5. Apply hardening to PB-MPM relaxation (based on solver iteration)
    float dynamic_relaxation = fminf(settings.relaxation * hardening, 1.0f);

    // 6. Update displacement
    Dp += diff * dynamic_relaxation;
}

__device__ void updateDeformationSnow(const MaterialSettings& settings, Matrix2f& Fp, Matrix2f& Fe, const Matrix2f Dp)
{
    // 1. Trial elastic deformation
    Matrix2f Fe_trial = (identity() + Dp) * Fe;

    // 2. SVD to decompose elastic trial matrix
    Matrix2f U, V;
    Vector2f Sigma;
    Fe_trial.svd(&U, &Sigma, &V);

    // 3. Disney Yield Condition: Clamp elastic singular values 
    Vector2f elasticSigma(
        fminf(fmaxf(Sigma.x, 1.0f - settings.crit_compression), 1.0f + settings.crit_stretch),
        fminf(fmaxf(Sigma.y, 1.0f - settings.crit_compression), 1.0f + settings.crit_stretch)
    );

    // 4. Update elastic deformation Fe
    Fe = U.diag_product(elasticSigma) * V.transpose();

    // 5. Accumulate yield excess into plastic deformation Fp
    Vector2f plasticRatio(
        Sigma.x / fmaxf(elasticSigma.x, 1e-6f),
        Sigma.y / fmaxf(elasticSigma.y, 1e-6f)
    );
    Matrix2f Fp_yield = V.diag_product(plasticRatio) * V.transpose();
    Matrix2f Fp_new = Fp_yield * Fp;

    // 6. Plastic volume limit safeguard
    float Jp_new = Fp_new.det();
    const float min_Jp = 0.2f;
    if (Jp_new < min_Jp) {
        float scale = sqrtf(min_Jp / fmaxf(Jp_new, 1e-6f));
        Fp_new *= scale;
    }

    Fp = Fp_new;
}
#pragma endregion

#pragma region Elastic
__device__ void computeDisplacementElastic(const MaterialSettings& settings, const Matrix2f Fe, Matrix2f& Dp)
{
    // Formula based on EA's paper is: D = Fe^-1 * A - I (in spatial space: D = A * Fe^-1)
    
    // 1. Compute trial deformation gradient F_trial = (I + Dp) * Fe
    Matrix2f F_trial = (identity() + Dp) * Fe;

    // 2. We need to find matrix A which is the closest matrix to F_trial with determinant = 1
    // 2.1 Shape preservation: extract rigid rotation target (A_shape) via SVD (safe Polar Decomposition)
    Matrix2f U, V;
    Vector2f Sigma;
    F_trial.svd(&U, &Sigma, &V);
    Matrix2f A_shape = U * V.transpose();
    // 2.2. Volume preservation: compute volume-preserving target (A_vol) with det == 1.0
    // A_vol = F/det(F) but that needs to be reestructured a little for n dimensions
    // s * det(F) = 1 so s = 1/det(F); s^2 * det(F) = 1 so s = 1/ sqrt(det(F)); s^3 * det(F) = 1 so s = 1 / cbrt(det(F))
    float df = F_trial.det();
    float sign = (df < 0.0f) ? -1.0f : 1.0f;
    float cdf = fminf(fmaxf(fabsf(df), 0.1f), 1000.0f);
    float scale = 1.0f / (sign * sqrtf(cdf));
    Matrix2f A_vol = scale * F_trial;
    // 2.3. Constraints are not orthogonal so we introduce an interpolating factor
    Matrix2f A = settings.elasticity_ratio * A_shape + (1.0f - settings.elasticity_ratio) * A_vol;

    // 3. Calculate target deformation and add the different to the displacement scaled by relaxation
    Matrix2f target_D = A * Fe.inverse() - identity();
    Dp += settings.relaxation * (target_D - Dp);
}

__device__ void updateDeformationElastic(const MaterialSettings& mat, Matrix2f& Fe, const Matrix2f Dp)
{
    // 1. Advance deformation gradient
    Matrix2f Fe_new = (identity() + Dp) * Fe;

    // 2. Use SVD to clamp values
    Matrix2f U, V;
    Vector2f Sigma;
    Fe_new.svd(&U, &Sigma, &V);

    Sigma.x = fminf(fmaxf(Sigma.x, 0.2f), 1000.0f);
    Sigma.y = fminf(fmaxf(Sigma.y, 0.2f), 1000.0f);

    Fe = U.diag_product(Sigma) * V.transpose();
}
#pragma endregion
#pragma endregion

#pragma region Collisions
__device__ void checkCollision(const Vector2f pos, CollisionObjectData obj, float& phi, Vector2f& n) 
{
    if (obj.type == 0) { // sphere
        Vector2f r = pos - obj.center;
        float dist = r.length();
        phi = dist - obj.size.x;
        n = (dist > 1e-5f) ? (r / dist) : Vector2f(0.0f, 1.0f);
    }
    else if (obj.type == 1) { // box
        Vector2f d_pos = pos - obj.center;

        if (obj.rotation != 0.0f) {
            float c = cosf(-obj.rotation);
            float s = sinf(-obj.rotation);
            float x_rot = c * d_pos.x - s * d_pos.y;
            float y_rot = s * d_pos.x + c * d_pos.y;
            d_pos = Vector2f(x_rot, y_rot);
        }

        Vector2f abs_pos(fabsf(d_pos.x), fabsf(d_pos.y));
        Vector2f q = abs_pos - obj.size;

        float outside_dist = Vector2f(fmaxf(q.x, 0.0f), fmaxf(q.y, 0.0f)).length();
        float inside_dist = fminf(fmaxf(q.x, q.y), 0.0f);
        phi = outside_dist + inside_dist;

        float sign_x = (d_pos.x < 0.0f) ? -1.0f : 1.0f;
        float sign_y = (d_pos.y < 0.0f) ? -1.0f : 1.0f;

        if (outside_dist > 0.0f) {
            n = Vector2f((q.x > 0.0f) ? sign_x * (q.x / outside_dist) : 0.0f,
                (q.y > 0.0f) ? sign_y * (q.y / outside_dist) : 0.0f);
        }
        else {
            if (q.x > q.y) n = Vector2f(sign_x, 0.0f);
            else n = Vector2f(0.0f, sign_y);
        }

        if (obj.rotation != 0.0f) {
            float c = cosf(obj.rotation);
            float s = sinf(obj.rotation);
            float nx = c * n.x - s * n.y;
            float ny = s * n.x + c * n.y;
            n = Vector2f(nx, ny);
        }
    }
}

__device__ void computeCollidersDisplacement(const Vector2f Xi, Vector2f& Di, CollisionManagerDeviceData colliders) {
    for (int i = 0; i < colliders.count; i++) {
        // 1. Compute candidate position 
        Vector2f Xi_pred = Xi + Di;

        // 2. Check collision
        CollisionObjectData obj = colliders.d_objects[i];
        float phi = 0.0f;
        Vector2f n(0.0f, 0.0f);

        checkCollision(Xi_pred, obj, phi, n);

        // 3. Recalculate displacement if predicted position goes inside object
        if (phi <= 0.0f) {
            float vn = Di.dot(n); // Normal displacement

            if (vn < 0.0f) {
                Vector2f Dt = Di - n * vn; // Tangential displacement
                float Dt_len = Dt.length();

                // Apply Coulomb friction to
                if (Dt_len > 1e-5f) {
                    float friction_limit = obj.friction * (-vn); // Friction scales with normal force

                    if (Dt_len <= friction_limit)
                        Dt = Vector2f(0.0f, 0.0f); // Static friction
                    else
                        Dt -= (Dt / Dt_len) * friction_limit; // Kinetic friction
                }

                // Reconstruct displacement: keep tangent, zero out inward normal
                Di = Dt;
            }
        }
    }
}

__device__ void pushOutOfCollider(Vector2f& Xp, Vector2f&Xp_delta, CollisionManagerDeviceData colliders) {
    for (int i = 0; i < colliders.count; i++) {
        // 1. Check collision
        CollisionObjectData obj = colliders.d_objects[i];
        float phi = 0.0f;
        Vector2f n(0.0f, 0.0f);

        checkCollision(Xp, obj, phi, n);

        // 2. Push out to surface
        if (phi < 0.0f) {
            float depth = -phi;

            // Push position out to surface
            Xp += n * depth;

            // Kill momentum 
            float vn = Xp_delta.dot(n);
            if (vn < 0.0f) {
                // Remove the normal velocity
                Xp_delta -= n * vn;

                // (Optional) Apply simple friction to tangential velocity
                Xp_delta *= 0.95f;
            }
        }
    }
}

#pragma endregion

#pragma region Solver
__global__ void solveConstraints_kernel(MaterialType d_mat, MaterialSettings d_settings, float* d_Jp, Matrix2f* d_Fp, Matrix2f* d_Fe, Matrix2f* d_Dp, const int num_particles)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= num_particles) return;

    switch (d_mat) {
    case MaterialType::WATER:
        computeDisplacementWater(d_settings, d_Jp[p], d_Dp[p]);
        break;
    case MaterialType::SNOW:
        computeDisplacementSnow(d_settings, d_Fp[p], d_Fe[p], d_Dp[p]);
        break;
    case MaterialType::ELASTIC:
        computeDisplacementElastic(d_settings, d_Fe[p], d_Dp[p]);
        break;
    }
}


__global__ void p2g_kernel(const Vector2f* d_Xp, const Vector2f* d_Xp_delta, const float* d_Mp, const Matrix2f* d_Dp, const int num_particles,
    float* d_Mi, Vector2f* d_Di, const int gridX, const int gridY)
{
    // (1) Get particle (thread per particle) and its characteristics
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= num_particles) return;

    Vector2f Xp = d_Xp[p];
    Vector2f Xp_delta = d_Xp_delta[p];
    Matrix2f Dp = d_Dp[p];
    float Mp = d_Mp[p];

    // (2) Find the bottom-left node closest to the particle of the 3x3 stencil (and init weights)
    Vector2f w[3], dw[3];
    Vector2f base = initQuadraticWeights(Xp, w, dw);

    // (3) Loop over the neighbor nodes
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {

            // (3.1) Get current node idx
            Vector2f node(base.x + x, base.y + y);
            int node_idx = node.x + (gridX + 1) * node.y;

            // (3.2) Compute accumulation on nodes

            // Offset from particle to node center is needed for APIC
            Vector2f offset = node - Xp;
            
            // Weighted mass: mi = sum(Wip * Mp)
            float Wip = w[x].x * w[y].y;
            float inMi = Wip * Mp;

            // Momemtum:
            Vector2f inDi = inMi * (Xp_delta + Dp * offset);

            // Atomic accumulation into GPU grid nodes
            atomicAdd(&d_Mi[node_idx], inMi);
            atomicAdd(&d_Di[node_idx].x, inDi.x);
            atomicAdd(&d_Di[node_idx].y, inDi.y);
        }
    }
}

__global__ void updateGrid_kernel(const float* d_Mi, Vector2f* d_Di,
    const int num_nodes, const int gridX, const int gridY, CollisionManagerDeviceData collisionData) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    float Mi = d_Mi[i];

    if (Mi < 1e-5f) {
        d_Di[i] = Vector2f(0.0f, 0.0f);
        return;
    }

    // Get grid displacement from momentum
    d_Di[i] /= Mi;

    // Update displacement with collisions
    int gx = i % (gridX + 1);
    int gy = i / (gridX + 1);
    Vector2f Xi ((float)gx, (float)gy);

    computeCollidersDisplacement(Xi, d_Di[i], collisionData);
}

__global__ void g2p_kernel(Vector2f* d_Xp, Vector2f* d_Xp_delta, Matrix2f* d_Dp, const int num_particles,
    Vector2f* d_Di, const int gridX, const int gridY)
{
    // (1) Get particle (thread per particle) and its characteristics
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= num_particles) return;

    Vector2f Xp = d_Xp[p];
    Vector2f Xp_delta(0.0f, 0.0f);
    Matrix2f Bp (0.0f, 0.0f, 0.0f, 0.0f);

    // (2) Get base grid node and init weights
    Vector2f w[3], dw[3];
    Vector2f base = initQuadraticWeights(Xp, w, dw);

    // (3) Stencil
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {
            // 3.1 Get current node idx
            Vector2f node(base.x + x, base.y + y);
            int node_idx = node.x + (gridX + 1) * node.y;

            // 3.2. Acumulate Bp and predicted position from weighted displacement
            float Wip = w[x].x * w[y].y;
            Vector2f WipDi = Wip * d_Di[node_idx];

            Xp_delta += WipDi;

            // Offset from particle to node center is needed for APIC
            Vector2f offset = node - Xp;
            Bp += outer_product(WipDi, offset);
        }
    }

    // (4) Write back the value 
    d_Dp[p] = Bp * 4.0f;
    d_Xp_delta[p] = Xp_delta;
}

__global__ void integrateParticle_kernel(Vector2f* d_Xp, Vector2f* d_Xp_delta, Matrix2f* d_Dp, 
    MaterialType mat, MaterialSettings settings, float* d_Jp, Matrix2f* d_Fp, Matrix2f* d_Fe,
    const int num_particles, const int gridX, const int gridY, const float dt, const float G, CollisionManagerDeviceData collisionData)
{
    // 1. Get particle (thread per particle) and its characteristics
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= num_particles) return;

    // 2. Add displacement to particle position
    d_Xp[p] += d_Xp_delta[p];

    // 3. Explicit external forces displacement (x = at^2)
    float gravityDisplacement = (float)gridY * G * dt * dt;
    d_Xp_delta[p].y -= gravityDisplacement;

    // 4. Check again against colliders to push position out of them if it is inside
    pushOutOfCollider(d_Xp[p], d_Xp_delta[p], collisionData);

    // 5. Clamp particle position so it isn't outside the domain of the grid 
    // because of the interpolation used and the cell being 1.0f wide the domain is 1.5f units less on each side than grid size
    d_Xp[p].x = fminf(fmaxf(d_Xp[p].x, 1.5f), (float)gridX - 1.5f);
    d_Xp[p].y = fminf(fmaxf(d_Xp[p].y, 1.5f), (float)gridY - 1.5f);

    // 6. Update deformation
    switch (mat) {
        case MaterialType::WATER:
            updateDeformationWater(settings, d_Jp[p], d_Dp[p]);
            break;
        case MaterialType::SNOW:
            updateDeformationSnow(settings, d_Fp[p], d_Fe[p], d_Dp[p]);
            break;
        case MaterialType::ELASTIC:
            updateDeformationElastic(settings, d_Fe[p], d_Dp[p]);
            break;
    }
}

#pragma endregion

#pragma region Host Solver Implementation

void solveConstraints(const ParticleSystem& ps)
{
    int blockSize = 256;
    int gridSize = (ps.num_particles + blockSize - 1) / blockSize;
    solveConstraints_kernel <<<gridSize, blockSize >>>
        (ps.type, ps.settings, ps.d_Jp, ps.d_Fp, ps.d_Fe, ps.d_Dp, ps.num_particles);
}

void p2g(const ParticleSystem& ps, Grid& grid)
{
    int blockSize = 256;
    int gridSize = (ps.num_particles + blockSize - 1) / blockSize;

    p2g_kernel <<<gridSize, blockSize >>> 
        (ps.d_Xp, ps.d_Xp_delta, ps.d_Mp, ps.d_Dp, ps.num_particles,
        grid.d_Mi, grid.d_Di, grid.grid_x, grid.grid_y);
}

void updateGrid(Grid& grid, CollisionManagerDeviceData collisionData)
{
    int blockSize = 256;
    int gridSize = (grid.num_nodes + blockSize - 1) / blockSize;

    updateGrid_kernel <<<gridSize, blockSize >>> 
        (grid.d_Mi, grid.d_Di, grid.num_nodes, grid.grid_x, grid.grid_y, collisionData);
}

void g2p(ParticleSystem& ps, const Grid& grid)
{
    int blockSize = 256;
    int gridSize = (ps.num_particles + blockSize - 1) / blockSize;

    g2p_kernel <<<gridSize, blockSize >>> 
        (ps.d_Xp, ps.d_Xp_delta, ps.d_Dp, ps.num_particles,
        grid.d_Di, grid.grid_x, grid.grid_y);
}

void integrateParticle(ParticleSystem& ps, const Grid& grid, float dt, float gravity, CollisionManagerDeviceData collisionData)
{
    if (ps.num_particles == 0) return;
    int blockSize = 256;
    int gridSize = (ps.num_particles + blockSize - 1) / blockSize;

    integrateParticle_kernel << <gridSize, blockSize >> >
        (ps.d_Xp, ps.d_Xp_delta, ps.d_Dp,
         ps.type, ps.settings, ps.d_Jp, ps.d_Fp, ps.d_Fe, ps.num_particles,
        grid.grid_x, grid.grid_y, dt, gravity, collisionData);
}

#pragma endregion
