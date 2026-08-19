#pragma once
#include <cuda_runtime.h>
#include "constants.h"
#include "types.h"
#include <vector>

class ParticleSystem {
public:
    int num_particles = 0;

    // --- GPU Device Pointers (allocated on GPU via cudaMalloc) ---
    float* d_Vp0 = nullptr; // Initial particle volume (constant)
    float* d_Mp = nullptr; // Particle mass (constant)

    Vector2f* d_Xp = nullptr; // Particle position
    Vector2f* d_Vp = nullptr; // Particle velocity
    Matrix2f* d_Bp = nullptr; // Particle APIC affine velocity field (2x2)

    // constitutive model (water for now) 
    float* Ap;												// For computation purpose
    float* Jp;												// Deformation gradient (det)

    void initialize(int count, const std::vector<Vector2f>& h_Xp, const std::vector<Vector2f>& h_Vp, float vp0 = 1.14f, float mp = 0.0005f) {
        num_particles = count;

        cudaMalloc(&d_Vp0, MAX_PARTICLES * sizeof(float));
        cudaMalloc(&d_Mp, MAX_PARTICLES * sizeof(float));
        cudaMalloc(&d_Xp, MAX_PARTICLES * sizeof(Vector2f));
        cudaMalloc(&d_Vp, MAX_PARTICLES * sizeof(Vector2f));
        cudaMalloc(&d_Bp, MAX_PARTICLES * sizeof(Matrix2f));
        cudaMalloc(&Ap, MAX_PARTICLES * sizeof(float));
        cudaMalloc(&Jp, MAX_PARTICLES * sizeof(float));

        if (num_particles > 0) {
            cudaMemcpy(d_Xp, h_Xp.data(), num_particles * sizeof(Vector2f), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Vp, h_Vp.data(), num_particles * sizeof(Vector2f), cudaMemcpyHostToDevice);

            std::vector<float> h_Mp(num_particles, mp);
            std::vector<float> h_Vp0(num_particles, vp0);
            std::vector<Matrix2f> h_Bp(num_particles, make_float4(0, 0, 0, 0));
            std::vector<float> h_Ap(num_particles, 0.0f);
            std::vector<float> h_Jp(num_particles, 1.0f);

            cudaMemcpy(d_Mp, h_Mp.data(), num_particles * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Vp0, h_Vp0.data(), num_particles * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Bp, h_Bp.data(), num_particles * sizeof(Matrix2f), cudaMemcpyHostToDevice);
            cudaMemcpy(Ap, h_Ap.data(), num_particles * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(Jp, h_Jp.data(), num_particles * sizeof(float), cudaMemcpyHostToDevice);
        }
    }

    void addParticlesMidSimulation(const std::vector<Vector2f>& new_pos, const std::vector<Vector2f>& new_vel, float vp0 = 1.14f, float mp = 0.0005f) {
        int add_count = static_cast<int>(new_pos.size());
        if (add_count == 0) return;

        // Prevent overflow
        if (num_particles + add_count >= MAX_PARTICLES) {
            add_count = std::max(0, MAX_PARTICLES - num_particles);
            std::cout << "\nReached particles limit. Cannot add more particles to simulation.\n";
        }

        int offset = num_particles;

        // Create new batch of vectors
        std::vector<float> h_Vp0(add_count, vp0);
        std::vector<float> h_Mp(add_count, mp);
        std::vector<Matrix2f> h_Bp(add_count, Matrix2f(0, 0, 0, 0));
        std::vector<float> h_Ap(add_count, 0.0f);
        std::vector<float> h_Jp(add_count, 1.0f);

        // Upload the new batch to memory, at the end of the last used position, on the reserved space
        cudaMemcpy(d_Xp + offset, new_pos.data(), add_count * sizeof(Vector2f), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Vp + offset, new_vel.data(), add_count * sizeof(Vector2f), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Vp0 + offset, h_Vp0.data(), add_count * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Mp + offset, h_Mp.data(), add_count * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Bp + offset, h_Bp.data(), add_count * sizeof(Matrix2f), cudaMemcpyHostToDevice);
        cudaMemcpy(Ap + offset, h_Ap.data(), add_count * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(Jp + offset, h_Jp.data(), add_count * sizeof(float), cudaMemcpyHostToDevice);

        // Update count
        num_particles += add_count;
    }

    void free() {
        cudaFree(d_Vp0); cudaFree(d_Mp);
        cudaFree(d_Xp);  cudaFree(d_Vp); cudaFree(d_Bp);
        cudaFree(Ap);    cudaFree(Jp);
    }
};