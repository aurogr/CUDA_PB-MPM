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
    //Vector2f* d_Vi = nullptr; // Node velocity
    Vector2f* d_Di = nullptr; // Displacement momemtum
    //Vector2f* d_Fi = nullptr; // Force applied to node // PB-MPM doesnt use internal forces

    void initialize(int res_x, int res_y) {
        grid_x = res_x;
        grid_y = res_y;
        num_nodes = (grid_x + 1) * (grid_y + 1);

        cudaMalloc(&d_Mi, num_nodes * sizeof(float));
        //cudaMalloc(&d_Vi, num_nodes * sizeof(Vector2f));
        //cudaMalloc(&d_Fi, num_nodes * sizeof(Vector2f));

        clear();
    }

    void clear() {
        cudaMemset(d_Mi, 0, num_nodes * sizeof(float));
        //cudaMemset(d_Vi, 0, num_nodes * sizeof(Vector2f));
        //cudaMemset(d_Fi, 0, num_nodes * sizeof(Vector2f));
    }

    void free() {
        cudaFree(d_Mi);
        //cudaFree(d_Vi); 
        //cudaFree(d_Fi);
    }
};