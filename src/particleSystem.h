#pragma once
#include <cuda_runtime.h>
#include "constants.h"
#include "types.h"
#include <vector>

struct WaterData {
    float* d_Jp = nullptr; // Deformation gradient determinant (volume change)

    const float RELAXATION = 0.85f; // Between 1.0f (perfectly incompressible) and 0.0f
                                    // even though water is incompressible we need to trade off some of it for stability

    void allocate(int num_particles) {
        cudaMalloc(&d_Jp, MAX_PARTICLES * sizeof(float));
        std::vector<float> h_Jp(num_particles, 1.0f);
        cudaMemcpy(d_Jp, h_Jp.data(), num_particles * sizeof(float), cudaMemcpyHostToDevice);
    }

    void addParticlesMidSimulation(int add_count, int offset) {
        std::vector<float> h_Jp(add_count, 1.0f);
        cudaMemcpy(d_Jp + offset, h_Jp.data(), add_count * sizeof(float), cudaMemcpyHostToDevice);
    }

    void free() {
        cudaFree(d_Jp);
    }
};

struct SnowData {

    Matrix2f* d_Fe; // Elastic deformation gradient
    Matrix2f* d_Fp; // Plastic deformation gradient
    float* d_Jp;    // Deformation gradient

    // CONSTANTS
    const float THT_C = 5.0e-3f;				// Critical compression
    const float THT_S = 5.0e-4f;				// Critical stretch (lower makes it brittle)
    const float KSI = 1.0f;						// Hardening coefficient
    const float RHO = 4.0e2f;					// Density
    const float E = 1.4e5f;						// Young's modulus
    const float V = 0.2f;						// Poisson's ratio

    const float MU_0 = E / (1.0f + V) / 2.0f;      // Base shear modulus
    const float LAMBDA_0 = E * V / (1.0f + V) / (1.0f - 2.0f * V);  // Base Lame's first parameter


    void allocate(int num_particles) {
        cudaMalloc(&d_Fe, MAX_PARTICLES * sizeof(Matrix2f));
        cudaMalloc(&d_Fp, MAX_PARTICLES * sizeof(Matrix2f));
        cudaMalloc(&d_Jp, MAX_PARTICLES * sizeof(float));

        std::vector<Matrix2f> h_Fe(num_particles, identity());
        std::vector<float> h_Jp(num_particles, 1.0f);

        cudaMemcpy(d_Fe, h_Fe.data(), num_particles * sizeof(Matrix2f), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Fp, h_Fe.data(), num_particles * sizeof(Matrix2f), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Jp, h_Jp.data(), num_particles * sizeof(float), cudaMemcpyHostToDevice);
    }
    void addParticlesMidSimulation(int add_count, int offset) const {

        // Create new batch of vectors
        std::vector<Matrix2f> h_Fe(add_count, identity());
        std::vector<float> h_Jp(add_count, 1.0);

        // Upload the new batch to memory, at the end of the last used position, on the reserved space
        cudaMemcpy(d_Fe + offset, h_Fe.data(), add_count * sizeof(Matrix2f), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Fp + offset, h_Fe.data(), add_count * sizeof(Matrix2f), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Jp + offset, h_Jp.data(), add_count * sizeof(float), cudaMemcpyHostToDevice);
    }
    void free() const {
        cudaFree(d_Fe);
        cudaFree(d_Jp);
    }
};

template <typename MatData>
class ParticleSystem {
public:
    int num_particles = 0;

    // --- GPU Device Pointers ---
    float* d_Mp = nullptr; // Particle mass (constant)

    Vector2f* d_Xp = nullptr; // Particle position
    Vector2f* d_Xp_delta = nullptr; // Particle position displacement
    Matrix2f* d_Dp = nullptr; // Particle deformation displacement

    // Material specific data is templated inside different structs
    MatData d_Mat;

    void initialize(int count, const std::vector<Vector2f>& h_Xp, const std::vector<Vector2f>& h_Xp_delta) {
        num_particles = count;

        cudaMalloc(&d_Mp, MAX_PARTICLES * sizeof(float));
        cudaMalloc(&d_Xp, MAX_PARTICLES * sizeof(Vector2f));
        cudaMalloc(&d_Xp_delta, MAX_PARTICLES * sizeof(Vector2f));
        cudaMalloc(&d_Dp, MAX_PARTICLES * sizeof(Matrix2f));

        if (num_particles > 0) {

            std::vector<float> h_Mp(num_particles, COMPUTED_MP0);
            std::vector<Matrix2f> h_Dp(num_particles, Matrix2f(0, 0, 0, 0));

            cudaMemcpy(d_Mp, h_Mp.data(), num_particles * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Xp, h_Xp.data(), num_particles * sizeof(Vector2f), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Xp_delta, h_Xp_delta.data(), num_particles * sizeof(Vector2f), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Dp, h_Dp.data(), num_particles * sizeof(Matrix2f), cudaMemcpyHostToDevice);
        }

        d_Mat.allocate(num_particles);
    }

    void addParticlesMidSimulation(const std::vector<Vector2f>& new_pos, const std::vector<Vector2f>& new_displacement) {
        int add_count = static_cast<int>(new_pos.size());
        if (add_count == 0) return;

        // Prevent overflow
        if (num_particles + add_count >= MAX_PARTICLES) {
            add_count = std::max(0, MAX_PARTICLES - num_particles);
            std::cout << "\nReached particles limit. Cannot add more particles to simulation.\n";
        }

        int offset = num_particles;

        // Create new batch of vectors
        std::vector<float> h_Mp(add_count, COMPUTED_MP0);
        std::vector<Matrix2f> h_Dp(add_count, Matrix2f(0, 0, 0, 0));

        // Upload the new batch to memory, at the end of the last used position, on the reserved space
        cudaMemcpy(d_Mp + offset, h_Mp.data(), add_count * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Xp + offset, new_pos.data(), add_count * sizeof(Vector2f), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Xp_delta + offset, new_displacement.data(), add_count * sizeof(Vector2f), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Dp + offset, h_Dp.data(), add_count * sizeof(Matrix2f), cudaMemcpyHostToDevice);

        // Update count
        num_particles += add_count;

        d_Mat.addParticlesMidSimulation(add_count, offset);
    }

    void free() {
        cudaFree(d_Mp);
        cudaFree(d_Xp);  
        cudaFree(d_Xp_delta); 
        cudaFree(d_Dp);

        d_Mat.free();
    }
};