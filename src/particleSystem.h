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
        cudaMalloc(&d_Vp0, num_particles * sizeof(float));
        cudaMalloc(&d_Mp, num_particles * sizeof(float));

        cudaMalloc(&d_Xp, num_particles * sizeof(float2));
        cudaMalloc(&d_Vp, num_particles * sizeof(float2));
        cudaMalloc(&d_Bp, num_particles * sizeof(float4));
        cudaMalloc(&Ap, num_particles * sizeof(float));
        cudaMalloc(&Jp, num_particles * sizeof(float));

        cudaMemcpy(d_Xp, h_Xp.data(), num_particles * sizeof(float2), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Vp, h_Vp.data(), num_particles * sizeof(float2), cudaMemcpyHostToDevice);

        std::vector<float> h_Mp(num_particles, 1.0f);
        std::vector<float4> h_Bp(num_particles, make_float4(0, 0, 0, 0));
        std::vector<float> h_Ap(num_particles, 0.0f);
        std::vector<float> h_Jp(num_particles, 1.0f);

        cudaMemcpy(d_Mp, h_Mp.data(), num_particles * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Bp, h_Bp.data(), num_particles * sizeof(float4), cudaMemcpyHostToDevice);
        cudaMemcpy(Ap, h_Ap.data(), num_particles * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(Jp, h_Jp.data(), num_particles * sizeof(float), cudaMemcpyHostToDevice);
    }

    void free() {
        cudaFree(d_Vp0); cudaFree(d_Mp);
        cudaFree(d_Xp);  cudaFree(d_Vp); cudaFree(d_Bp);
        cudaFree(Ap);    cudaFree(Jp);
    }
};