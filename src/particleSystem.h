#pragma once
#include <cuda_runtime.h>
#include "constants.h"
#include <vector>

class ParticleSystem {
public:
    int num_particles = 0;

    // --- GPU Device Pointers (allocated on GPU via cudaMalloc) ---
    float* d_Vp0 = nullptr; // Initial particle volume (constant)
    float* d_Mp = nullptr; // Particle mass (constant)

    float2* d_Xp = nullptr; // Particle position
    float2* d_Vp = nullptr; // Particle velocity
    float4* d_Bp = nullptr; // Particle APIC affine velocity field (2x2)

    // constitutive model (water for now) 
    float* Ap;												// For computation purpose
    float* Jp;												// Deformation gradient (det)

    void initialize(int count, const std::vector<float2>& h_Xp, const std::vector<float2>& h_Vp) {
        num_particles = count;

        cudaMalloc(&d_Vp0, MAX_PARTICLES * sizeof(float));
        cudaMalloc(&d_Mp, MAX_PARTICLES * sizeof(float));
        cudaMalloc(&d_Xp, MAX_PARTICLES * sizeof(float2));
        cudaMalloc(&d_Vp, MAX_PARTICLES * sizeof(float2));
        cudaMalloc(&d_Bp, MAX_PARTICLES * sizeof(float4));
        cudaMalloc(&Ap, MAX_PARTICLES * sizeof(float));
        cudaMalloc(&Jp, MAX_PARTICLES * sizeof(float));

        if (num_particles > 0) {
            cudaMemcpy(d_Xp, h_Xp.data(), num_particles * sizeof(float2), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Vp, h_Vp.data(), num_particles * sizeof(float2), cudaMemcpyHostToDevice);

            // Initial material properties (Water baseline)
            float particle_mass = 0.0005f;
            float p_vol = 1.14f; // Initial particle volume estimate in 2D cell

            std::vector<float> h_Mp(num_particles, particle_mass);
            std::vector<float> h_Vp0(num_particles, p_vol);
            std::vector<float4> h_Bp(num_particles, make_float4(0, 0, 0, 0));
            std::vector<float> h_Ap(num_particles, 0.0f);
            std::vector<float> h_Jp(num_particles, 1.0f);

            cudaMemcpy(d_Mp, h_Mp.data(), num_particles * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Vp0, h_Vp0.data(), num_particles * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Bp, h_Bp.data(), num_particles * sizeof(float4), cudaMemcpyHostToDevice);
            cudaMemcpy(Ap, h_Ap.data(), num_particles * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(Jp, h_Jp.data(), num_particles * sizeof(float), cudaMemcpyHostToDevice);
        }
    }

    void addParticlesMidSimulation(const std::vector<float2>& new_pos, const std::vector<float2>& new_vel, float vp0 = 1.14f, float mp = 0.0005f) {
        int add_count = static_cast<int>(new_pos.size());
        if (add_count == 0) return;

        // Prevent overflow beyond pre-allocated bound
        if (num_particles + add_count >= MAX_PARTICLES) {
            add_count = std::max(0, MAX_PARTICLES - num_particles);
            std::cout << "\nReached particles limit. Cannot add more particles to simulation.\n";
        }

        int offset = num_particles;

        // Prepare initial state vectors for new batch
        std::vector<float> h_Vp0(add_count, vp0);
        std::vector<float> h_Mp(add_count, mp);
        std::vector<float4> h_Bp(add_count, make_float4(0, 0, 0, 0));
        std::vector<float> h_Ap(add_count, 0.0f);
        std::vector<float> h_Jp(add_count, 1.0f);

        // Copy directly to the end of active data (pointer arithmetic: d_Xp + offset)
        cudaMemcpy(d_Xp + offset, new_pos.data(), add_count * sizeof(float2), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Vp + offset, new_vel.data(), add_count * sizeof(float2), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Vp0 + offset, h_Vp0.data(), add_count * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Mp + offset, h_Mp.data(), add_count * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Bp + offset, h_Bp.data(), add_count * sizeof(float4), cudaMemcpyHostToDevice);
        cudaMemcpy(Ap + offset, h_Ap.data(), add_count * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(Jp + offset, h_Jp.data(), add_count * sizeof(float), cudaMemcpyHostToDevice);

        // Simply bump the active count
        num_particles += add_count;
    }

    void free() {
        cudaFree(d_Vp0); cudaFree(d_Mp);
        cudaFree(d_Xp);  cudaFree(d_Vp); cudaFree(d_Bp);
        cudaFree(Ap);    cudaFree(Jp);
    }
};