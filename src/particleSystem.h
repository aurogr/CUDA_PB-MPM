#pragma once
#include <cuda_runtime.h>
#include "constants.h"
#include "types.h"
#include <vector>

struct WaterData {
    // constitutive model (water for now) 
    float* d_Ap = nullptr; // For computation purpose
    float* d_Jp = nullptr; // Deformation gradient (det)
    
    // constants for material (they could also be con constants.h)
    const float RHO = 1.0;					// Density
    const float K = 50.0;					// Bulk Modulus
    const int   GAMMA = 3;					// Penalize deviation form incompressibility

    void allocate(int num_particles) {
        cudaMalloc(&d_Ap, MAX_PARTICLES * sizeof(float));
        cudaMalloc(&d_Jp, MAX_PARTICLES * sizeof(float));

        std::vector<float> h_Jp(num_particles, 1.0f);

        cudaMemset(d_Ap, 0, num_particles * sizeof(float));
        cudaMemcpy(d_Jp, h_Jp.data(), num_particles * sizeof(float), cudaMemcpyHostToDevice);
    }

    void addParticlesMidSimulation(int add_count, int offset) {

        // Create new batch of vectors
        std::vector<float> h_Jp(add_count, 1.0f);

        // Upload the new batch to memory, at the end of the last used position, on the reserved space
        cudaMemset(d_Ap + offset, 0, add_count * sizeof(float));
        cudaMemcpy(d_Jp + offset, h_Jp.data(), add_count * sizeof(float), cudaMemcpyHostToDevice);
    }

    void free() {
        cudaFree(d_Ap);
        cudaFree(d_Jp);
    }
};

struct SnowData {

    Matrix2f* d_Fe; // Elastic deformation gradient
    float* d_Jp;    // Plastic volume determinant

    // CONSTANTS
    const double THT_C = 2.0e-2;				// Critical compression
    const double THT_S = 6.0e-3;				// Critical stretch
    const double KSI = 10;						// Hardening coefficient
    const double RHO = 4.0e2;					// Density
    const double E = 1.4e5;						// Young's modulus
    const double V = 0.2;						// Poisson's ratio

    const float MU_0 = E / (1.0 + V) / 2.0;      // Base shear modulus
    const float LAMBDA_0 = E * V / (1.0 + V) / (1.0 - 2.0 * V);  // Base Lame's first parameter


    void allocate(int num_particles) {
        cudaMalloc(&d_Fe, MAX_PARTICLES * sizeof(Matrix2f));
        cudaMalloc(&d_Jp, MAX_PARTICLES * sizeof(float));

        std::vector<Matrix2f> h_Fe(num_particles, identity());
        std::vector<float> h_Jp(num_particles, 1.0);

        cudaMemcpy(d_Fe, h_Fe.data(), num_particles * sizeof(Matrix2f), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Jp, h_Jp.data(), num_particles * sizeof(float), cudaMemcpyHostToDevice);
    }
    void addParticlesMidSimulation(int add_count, int offset) {

        // Create new batch of vectors
        std::vector<Matrix2f> h_Fe(add_count, identity());
        std::vector<float> h_Jp(add_count, 1.0);

        // Upload the new batch to memory, at the end of the last used position, on the reserved space
        cudaMemcpy(d_Fe + offset, h_Fe.data(), add_count * sizeof(Matrix2f), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Jp + offset, h_Jp.data(), add_count * sizeof(float), cudaMemcpyHostToDevice);
    }
    void free() {
        cudaFree(d_Fe);
        cudaFree(d_Jp);
    }
};

template <typename MatData>
class ParticleSystem {
public:
    int num_particles = 0;

    // --- GPU Device Pointers ---
    float* d_Vp0 = nullptr; // Initial particle volume (constant)
    float* d_Mp = nullptr; // Particle mass (constant)

    Vector2f* d_Xp = nullptr; // Particle position
    Vector2f* d_Vp = nullptr; // Particle velocity
    Matrix2f* d_Bp = nullptr; // Particle APIC affine velocity field (2x2)

    // Material specific data is templated inside different structs
    MatData d_Mat;

    void initialize(int count, const std::vector<Vector2f>& h_Xp, const std::vector<Vector2f>& h_Vp, float vp0 = 1.14f, float mp = 0.0005f) {
        num_particles = count;

        cudaMalloc(&d_Vp0, MAX_PARTICLES * sizeof(float));
        cudaMalloc(&d_Mp, MAX_PARTICLES * sizeof(float));
        cudaMalloc(&d_Xp, MAX_PARTICLES * sizeof(Vector2f));
        cudaMalloc(&d_Vp, MAX_PARTICLES * sizeof(Vector2f));
        cudaMalloc(&d_Bp, MAX_PARTICLES * sizeof(Matrix2f));

        if (num_particles > 0) {

            std::vector<float> h_Mp(num_particles, mp);
            std::vector<float> h_Vp0(num_particles, vp0);

            cudaMemcpy(d_Xp, h_Xp.data(), num_particles * sizeof(Vector2f), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Vp, h_Vp.data(), num_particles * sizeof(Vector2f), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Mp, h_Mp.data(), num_particles * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Vp0, h_Vp0.data(), num_particles * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemset(d_Bp, 0, num_particles * sizeof(Matrix2f));
        }

        d_Mat.allocate(num_particles);
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

        // Upload the new batch to memory, at the end of the last used position, on the reserved space
        cudaMemcpy(d_Xp + offset, new_pos.data(), add_count * sizeof(Vector2f), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Vp + offset, new_vel.data(), add_count * sizeof(Vector2f), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Vp0 + offset, h_Vp0.data(), add_count * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Mp + offset, h_Mp.data(), add_count * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Bp + offset, h_Bp.data(), add_count * sizeof(Matrix2f), cudaMemcpyHostToDevice);

        // Update count
        num_particles += add_count;

        d_Mat.addParticlesMidSimulation(add_count, offset);
    }

    void free() {
        cudaFree(d_Vp0); cudaFree(d_Mp);
        cudaFree(d_Xp);  cudaFree(d_Vp); cudaFree(d_Bp);

        d_Mat.free();
    }
};