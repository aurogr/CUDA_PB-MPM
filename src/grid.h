#pragma once
#include <cuda_runtime.h>
#include <GLFW/glfw3.h>

#include "constants.h"
#include "types.h"

class Grid {
public:
    int grid_x = 0;
    int grid_y = 0;
    int num_nodes = 0;
    // grid spacing is going to be 1 for simplicity

    // --- GPU Device Pointers ---
    float* d_Mi = nullptr; // Node mass
    Vector2f* d_Di = nullptr; // Displacement momemtum

    void initialize(int res_x, int res_y) {
        grid_x = res_x;
        grid_y = res_y;
        num_nodes = (grid_x + 1) * (grid_y + 1);

        cudaMalloc(&d_Mi, num_nodes * sizeof(float));
        cudaMalloc(&d_Di, num_nodes * sizeof(Matrix2f));

        clear();
    }

    void clear() {
        cudaMemset(d_Mi, 0, num_nodes * sizeof(float));
        cudaMemset(d_Di, 0, num_nodes * sizeof(Matrix2f));
    }

    void free() {
        cudaFree(d_Mi);
    }
};