#pragma once
#include <cuda_runtime.h>
#include "constants.h"

class Grid {
public:
    int grid_x = 0;
    int grid_y = 0;
    int num_nodes = 0;
    float grid_spacing = 1.0f; // 'h' in literature

    // --- GPU Device Pointers ---
    float* d_Mi = nullptr; // Node mass
    float2* d_Xi = nullptr; // Node position
    float2* d_Vi = nullptr; // Node velocity
    float2* d_Fi = nullptr; // Force applied to node

    void initialize(int res_x, int res_y) {
        grid_x = res_x;
        grid_y = res_y;
        num_nodes = grid_x * grid_y;

        cudaMalloc(&d_Mi, num_nodes * sizeof(float));
        cudaMalloc(&d_Xi, num_nodes * sizeof(float2));
        cudaMalloc(&d_Vi, num_nodes * sizeof(float2));
        cudaMalloc(&d_Fi, num_nodes * sizeof(float2));
    }

    void free() {
        cudaFree(d_Mi); cudaFree(d_Xi);
        cudaFree(d_Vi); cudaFree(d_Fi);
    }
};