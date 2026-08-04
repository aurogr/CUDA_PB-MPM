#pragma once
#include <cuda_runtime.h>

#include "particleSystem.h"
#include "grid.h"
#include "constants.h"

void run_p2g(const ParticleSystem& ps, Grid& grid, float h);

/// <summary>
/// Init quadratic weights (grid step size is assumed to be 1.0)
/// </summary>
/// <param name="Xp">Particle position</param>
/// <param name="w">Returns the 3 stencil weights</param>
/// <param name="dw">Returns the 3 stencil weights derivatives>
/// <returns>Starting node index for the 3-node stencil</returns>
__device__ inline int InitQuadraticWeights(float Xp, float w[3], float dw[3])
{
    float flooredPos = floorf(Xp);
    float offset = (Xp - flooredPos) - 0.5f;

    w[0] = 0.5f * (0.5f - offset) * (0.5f - offset);
    w[1] = 0.75f - (offset * offset);
    w[2] = 0.5f * (0.5f + offset) * (0.5f + offset);

    dw[0] = offset - 0.5f;
    dw[1] = -2.0f * offset;
    dw[2] = offset + 0.5f;

    return static_cast<int>(flooredPos) - 1;
}
