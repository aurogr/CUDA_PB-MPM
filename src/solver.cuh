#pragma once
#include <cuda_runtime.h>

#include "particleSystem.h"
#include "grid.h"
#include "constants.h"
#include "boundary.h"

/// <summary>
/// Particle to grid. Transfer particle mass, momentum, and force (computed using stresses) to grid nodes using interpolation functions.
/// </summary>
template <typename MatData>
void p2g(const ParticleSystem<MatData>& ps, Grid& grid, float dt);

/// <summary>
/// Update nodal velocities via time integration while enforcing boundary conditions.
/// </summary>
void updateGrid(Grid& grid, float dt, CollisionManagerData collisionData);

/// <summary>
/// Interpolate updated grid velocities back to particles. Update each particle’s deformation gradient based on local velocity gradients and advect particle positions using updated velocities.
/// </summary>
template <typename MatData>
void g2p(ParticleSystem<MatData>& ps, const Grid& grid, float dt);

/// <summary>
/// Init quadratic weights (grid step size is assumed to be 1.0)
/// </summary>
/// <param name="Xp">Particle position</param>
/// <param name="w">Returns the 3 stencil weights</param>
/// <param name="dw">Returns the 3 stencil weights derivatives>
/// <returns>Starting node index for the 3-node stencil</returns>
__device__ inline Vector2f InitQuadraticWeights(Vector2f Xp, Vector2f w[3], Vector2f dw[3])
{
    // Quadratic stencil base node starts at floor(Xp - 0.5f)
    Vector2f base_node(
        floorf(Xp.x - 0.5f),
        floorf(Xp.y - 0.5f)
    );

    // Offset from the base node center
    Vector2f offset = Xp - base_node;

    // B-spline weights 
    Vector2f d0 = Vector2f(1.5f, 1.5f) - offset;
    Vector2f d1 = offset - Vector2f(1.0f, 1.0f);
    Vector2f d2 = offset - Vector2f(0.5f, 0.5f);

    w[0] = Vector2f(0.5f * d0.x * d0.x, 0.5f * d0.y * d0.y);
    w[1] = Vector2f(0.75f - d1.x * d1.x, 0.75f - d1.y * d1.y);
    w[2] = Vector2f(0.5f * d2.x * d2.x, 0.5f * d2.y * d2.y);

    // Gradients
    dw[0] = offset - Vector2f(1.5f, 1.5f);
    dw[1] = (offset - Vector2f(1.0f, 1.0f)) * -2.0f;
    dw[2] = offset - Vector2f(0.5f, 0.5f);

    return base_node;
}
