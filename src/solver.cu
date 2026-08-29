#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "solver.cuh"
#include "boundary.h"
#include "types.h"


#pragma region Material
__device__ void computeDisplacement(const WaterData& mat, int p, Matrix2f& Dp) 
{
    // (1) If we wanted liquids with some viscosity we would need to add a step, for now we only have water
    
    // (2) Volume preservation (Incompressibility)
    
    // Liquids are incompressible, their volume ratio J (det(F)) needs to equal 1.0f
    // we need to calculate an impulse (alpha) that forces the liquid back to its resting volume
    
    // The two identities used for that are:
    // 1. det(F_new) = 1.0f;
    // 2. det(F_new) = det(I + D_p) * det(F_old); where we can linearize the determinant as: det(I + D_p) = 1.0f + Tr(D_p) 
    
    float alpha = 0.5f * (1.0f / mat.d_Jp[p] - Dp.trace() - 1.0f); // where the 0.5f comes from using an identity matrix in 2d space so the trace increases by 2*alpha

    // Finally, we add to the deformation displacement towards preserving the volume
    Dp += mat.RELAXATION * alpha * identity();
}

__device__ void updateDeformation(const WaterData& mat, int p, Matrix2f Dp) {
    // Liquids hold no memory of shape, they only care about volume, so storing only the determinant of the deformation gradient is enough

    // The true update for volume is det(F_new) = det(I + Dp) * det(F_old)
    // but as before, we can use the trace instead of the complicated determinant
    mat.d_Jp[p] *= (1.0f + Dp.trace());
    mat.d_Jp[p] = fmaxf(mat.d_Jp[p], 0.1f); // Never allow J <= 0
}

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

__device__ void computeCollidersDisplacement(const Vector2f Xi, Vector2f& Di, CollisionManagerData colliders) {
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

__device__ void pushOutOfCollider(Vector2f& Xp, Vector2f&Xp_delta, CollisionManagerData colliders) {
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
                // Xp_delta *= 0.95f;
            }
        }
    }
}

#pragma endregion

#pragma region Solver
template <typename MatData>
__global__ void solveConstraints_kernel(MatData d_mat, Matrix2f* d_Dp, const int num_particles)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= num_particles) return;

    computeDisplacement(d_mat, p, d_Dp[p]);
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
    const int num_nodes, const int gridX, const int gridY, CollisionManagerData collisionData) 
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

template <typename MatData>
__global__ void integrateParticle_kernel(Vector2f* d_Xp, Vector2f* d_Xp_delta, Matrix2f* d_Dp, MatData d_mat,
    const int num_particles, const int gridX, const int gridY, const float dt, const float G, CollisionManagerData collisionData)
{
    // 1. Get particle (thread per particle) and its characteristics
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= num_particles) return;

    // 2. Add displacement to particle position
    d_Xp[p] += d_Xp_delta[p];

    // 3. Explicit external forces displacement (x = at^2)
    d_Xp_delta[p].y -= G * dt * dt;

    // 4. Check again against colliders to push position out of them if it is inside
    pushOutOfCollider(d_Xp[p], d_Xp_delta[p], collisionData);

    // 5. Clamp particle position so it isn't outside the domain of the grid 
    // because of the interpolation used and the cell being 1.0f wide the domain is 1.5f units less on each side than grid size
    d_Xp[p].x = fminf(fmaxf(d_Xp[p].x, 1.5f), (float)gridX - 1.5f);
    d_Xp[p].y = fminf(fmaxf(d_Xp[p].y, 1.5f), (float)gridY - 1.5f);

    // 6. Update deformation
    updateDeformation(d_mat, p, d_Dp[p]);
}

#pragma endregion

#pragma region Host Solver Implementation

template <typename MatData>
void solveConstraints(const ParticleSystem<MatData>& ps)
{
    int blockSize = 256;
    int gridSize = (ps.num_particles + blockSize - 1) / blockSize;
    solveConstraints_kernel <<<gridSize, blockSize >>>
        (ps.d_Mat, ps.d_Dp, ps.num_particles);
}

template <typename MatData>
void p2g(const ParticleSystem<MatData>& ps, Grid& grid)
{
    int blockSize = 256;
    int gridSize = (ps.num_particles + blockSize - 1) / blockSize;

    p2g_kernel <<<gridSize, blockSize >>> 
        (ps.d_Xp, ps.d_Xp_delta, ps.d_Mp, ps.d_Dp, ps.num_particles,
        grid.d_Mi, grid.d_Di, grid.grid_x, grid.grid_y);
}

void updateGrid(Grid& grid, CollisionManagerData collisionData)
{
    int blockSize = 256;
    int gridSize = (grid.num_nodes + blockSize - 1) / blockSize;

    updateGrid_kernel <<<gridSize, blockSize >>> 
        (grid.d_Mi, grid.d_Di, grid.num_nodes, grid.grid_x, grid.grid_y, collisionData);
}

template <typename MatData>
void g2p(ParticleSystem<MatData>& ps, const Grid& grid)
{
    int blockSize = 256;
    int gridSize = (ps.num_particles + blockSize - 1) / blockSize;

    g2p_kernel <<<gridSize, blockSize >>> 
        (ps.d_Xp, ps.d_Xp_delta, ps.d_Dp, ps.num_particles,
        grid.d_Di, grid.grid_x, grid.grid_y);
}

template <typename MatData>
void integrateParticle(ParticleSystem<MatData>& ps, const Grid& grid, float dt, CollisionManagerData collisionData)
{
    if (ps.num_particles == 0) return;
    int blockSize = 256;
    int gridSize = (ps.num_particles + blockSize - 1) / blockSize;

    integrateParticle_kernel << <gridSize, blockSize >> >
        (ps.d_Xp, ps.d_Xp_delta, ps.d_Dp, ps.d_Mat, ps.num_particles,
        grid.grid_x, grid.grid_y, dt, GRAVITY, collisionData);
}

#pragma endregion

#pragma region Explicit instantiation
template void solveConstraints<WaterData>(const ParticleSystem<WaterData>& ps);

template void p2g<WaterData>(const ParticleSystem<WaterData>& ps, Grid& grid);

template void g2p<WaterData>(ParticleSystem<WaterData>& ps, const Grid& grid);

template void integrateParticle<WaterData>(ParticleSystem<WaterData>& ps, const Grid& grid, float dt, CollisionManagerData collisionData);

#pragma endregion