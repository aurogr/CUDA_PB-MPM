#pragma once
#include <cuda_runtime.h>
#include "constants.h"
#include "types.h"
#include <vector>

enum class MaterialType {
    WATER,
    SNOW,
    ELASTIC
};

struct MaterialSettings {
    // Shared
    float relaxation = 0.9f; // water, snow and elastic

    // Water specific
    float viscosity = 0.0f;

    // Snow specific
    float crit_compression = 0.025f;
    float crit_stretch = 0.025f;
    float hard_coeff = 10.0f;

    // Elastic specific
    float elasticity_ratio = 0.9f;
};

class ParticleSystem {
public:
    int num_particles = 0;
    MaterialType type;

    // --- UI Editable Settings ---
    MaterialSettings settings;

    // --- Common GPU Pointers ---
    float* d_Mp = nullptr;
    Vector2f* d_Xp = nullptr;
    Vector2f* d_Xp_delta = nullptr;
    Matrix2f* d_Dp = nullptr;

    // --- Material-Specific GPU Pointers ---
    float* d_Jp = nullptr;       // Water
    Matrix2f* d_Fe = nullptr;    // Snow & Elastic
    Matrix2f* d_Fp = nullptr;    // Snow

    ParticleSystem(MaterialType matType) {
        type = matType;
    }

    ~ParticleSystem() {
        free();
    }

    void allocate() {

        cudaMalloc(&d_Mp, MAX_PARTICLES * sizeof(float));
        cudaMalloc(&d_Xp, MAX_PARTICLES * sizeof(Vector2f));
        cudaMalloc(&d_Xp_delta, MAX_PARTICLES * sizeof(Vector2f));
        cudaMalloc(&d_Dp, MAX_PARTICLES * sizeof(Matrix2f));

        switch (type) {
        case MaterialType::WATER:
            cudaMalloc(&d_Jp, MAX_PARTICLES * sizeof(float));
            break;
        case MaterialType::SNOW:
            cudaMalloc(&d_Fp, MAX_PARTICLES * sizeof(Matrix2f));
            cudaMalloc(&d_Fe, MAX_PARTICLES * sizeof(Matrix2f));
            break;
        case MaterialType::ELASTIC:
            cudaMalloc(&d_Fe, MAX_PARTICLES * sizeof(Matrix2f));
            break;
        }
    }

    void initialize(int count, const std::vector<Vector2f>& h_Xp, const std::vector<Vector2f>& h_Xp_delta) {
        num_particles = count;

        if (num_particles > 0) {

            std::vector<float> h_Mp(num_particles, COMPUTED_MP0);
            std::vector<Matrix2f> h_Dp(num_particles, Matrix2f(0, 0, 0, 0));

            cudaMemcpy(d_Mp, h_Mp.data(), num_particles * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Xp, h_Xp.data(), num_particles * sizeof(Vector2f), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Xp_delta, h_Xp_delta.data(), num_particles * sizeof(Vector2f), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Dp, h_Dp.data(), num_particles * sizeof(Matrix2f), cudaMemcpyHostToDevice);

            switch (type) {
                case MaterialType::WATER: {
                    std::vector<float> h_Jp(num_particles, 1.0f);
                    cudaMemcpy(d_Jp, h_Jp.data(), num_particles * sizeof(float), cudaMemcpyHostToDevice);
                    break;
                }
                case MaterialType::SNOW: {
                    std::vector<Matrix2f> h_F(num_particles, identity());
                    cudaMemcpy(d_Fp, h_F.data(), num_particles * sizeof(Matrix2f), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_Fe, h_F.data(), num_particles * sizeof(Matrix2f), cudaMemcpyHostToDevice);
                    break;
                }
                case MaterialType::ELASTIC: {
                    std::vector<Matrix2f> h_Fe(num_particles, identity());
                    cudaMemcpy(d_Fe, h_Fe.data(), num_particles * sizeof(Matrix2f), cudaMemcpyHostToDevice);
                    break;
                }
            }
        }
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

        switch (type) {
        case MaterialType::WATER: {
            std::vector<float> h_Jp(add_count, 1.0f);
            cudaMemcpy(d_Jp + offset, h_Jp.data(), add_count * sizeof(float), cudaMemcpyHostToDevice);
            break;
        }
        case MaterialType::SNOW: {
            std::vector<Matrix2f> h_F(add_count, identity());
            cudaMemcpy(d_Fp + offset, h_F.data(), add_count * sizeof(Matrix2f), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Fe + offset, h_F.data(), add_count * sizeof(Matrix2f), cudaMemcpyHostToDevice);
            break;
        }
        case MaterialType::ELASTIC: {
            std::vector<Matrix2f> h_Fe(add_count, identity());
            cudaMemcpy(d_Fe + offset, h_Fe.data(), add_count * sizeof(Matrix2f), cudaMemcpyHostToDevice);
            break;
        }
        }
    }

    void free() {
        cudaFree(d_Mp);
        cudaFree(d_Xp);  
        cudaFree(d_Xp_delta); 
        cudaFree(d_Dp);
        cudaFree(d_Jp);
        cudaFree(d_Fp);
        cudaFree(d_Fe);
    }
};

struct SimulationParticles
{
    ParticleSystem water = ParticleSystem(MaterialType::WATER);
    ParticleSystem snow = ParticleSystem(MaterialType::SNOW);
    ParticleSystem elastic = ParticleSystem(MaterialType::ELASTIC);

    int inline getParticlesCount() {
        return water.num_particles + snow.num_particles + elastic.num_particles;
    }

    void inline free() {
        water.free();
        snow.free();
        elastic.free();
    }
};