#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "constants.h"
#include "solver.cuh"
#include "grid.h"
#include "particleSystem.h"


int main() {
    constexpr int NUM_PARTICLES = 1000;

    std::cout << "Initializing CUDA MPM2D Simulation...\n";
    std::cout << "Particles: " << NUM_PARTICLES << " | Grid: " << X_GRID << "x" << Y_GRID << "\n";

        // 1. Initialize Host (CPU) data using standard float2
        std::vector<float2> h_Xp(NUM_PARTICLES);
    std::vector<float2> h_Vp(NUM_PARTICLES);

    // Spawn particles inside a 2D box (e.g., [2,2] to [8,8])
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        float fx = 2.0f + static_cast<float>(rand() % 600) / 100.0f;
        float fy = 2.0f + static_cast<float>(rand() % 600) / 100.0f;

        h_Xp[i] = make_float2(fx, fy);
        h_Vp[i] = make_float2(0.0f, -9.8f); // Initial downward velocity
    }

    // 2. Initialize GPU Grid and Particle System
    Grid grid;
    grid.initialize(X_GRID, Y_GRID);

        ParticleSystem particle_system;
    particle_system.initialize(NUM_PARTICLES, h_Xp, h_Vp); 

        std::cout << "GPU Memory successfully allocated and copied!\n";

        // 3. Test Kernel Launch
        float cell_spacing = 1.0f;
        std::cout << "Launching P2G Kernel...\n";
        run_p2g(particle_system, grid, cell_spacing);

        // 4. Verify GPU Result (Copy back node mass array to CPU)
        std::vector<float> h_Mi(grid.num_nodes);
        cudaMemcpy(h_Mi.data(), grid.d_Mi, grid.num_nodes * sizeof(float), cudaMemcpyDeviceToHost);

            float total_grid_mass = 0.0f;
        for (float mass : h_Mi) {
            total_grid_mass += mass;
        }

        std::cout << "P2G Success! Total Mass transferred to Grid: "
            << total_grid_mass << " (Expected ~" << NUM_PARTICLES << ".0)\n";

        // 5. Cleanup
        particle_system.free(); 
            grid.free(); 

            std::cout << "Simulation trial finished cleanly.\n";
        return 0;
}