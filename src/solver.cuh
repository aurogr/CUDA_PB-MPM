#pragma once
#include <cuda_runtime.h>

#include "particleSystem.h"
#include "grid.h"
#include "constants.h"
#include "boundary.h"

/// <summary>
/// Particle to grid. Transfer particle mass, momentum, and force (computed using stresses) to grid nodes using interpolation functions.
/// </summary>
void p2g(const ParticleSystem& ps, Grid& grid, float dt);

/// <summary>
/// Update nodal velocities via time integration while enforcing boundary conditions.
/// </summary>
void updateGrid(Grid& grid, float dt, BoundaryData bounds);

/// <summary>
/// Interpolate updated grid velocities back to particles. Update each particle’s deformation gradient based on local velocity gradients and advect particle positions using updated velocities.
/// </summary>
void g2p(ParticleSystem& ps, const Grid& grid, float dt);

/// <summary>
/// Init quadratic weights (grid step size is assumed to be 1.0)
/// </summary>
/// <param name="Xp">Particle position</param>
/// <param name="w">Returns the 3 stencil weights</param>
/// <param name="dw">Returns the 3 stencil weights derivatives>
/// <returns>Starting node index for the 3-node stencil</returns>
__device__ inline int InitQuadraticWeights(float Xp, float w[3], float dw[3])
{
    // Quadratic stencil base node starts at floor(Xp - 0.5f)
    int base_node = static_cast<int>(floorf(Xp - 0.5f));

    // Offset from the base node center
    float offset = Xp - static_cast<float>(base_node);

    // B-spline weights 
    w[0] = 0.5f * (1.5f - offset) * (1.5f - offset);
    w[1] = 0.75f - (offset - 1.0f) * (offset - 1.0f);
    w[2] = 0.5f * (offset - 0.5f) * (offset - 0.5f);

    // gradients of weights (d/dXp)
    dw[0] = offset - 1.5f;
    dw[1] = -2.0f * (offset - 1.0f);
    dw[2] = offset - 0.5f;

    return base_node;
}
