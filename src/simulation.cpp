#include <cuda_runtime.h>

#include <random>

#include "simulation.h"
#include "constants.h"
#include "solver.cuh"

Simulation::Simulation() {
}

Simulation::~Simulation() {
    free();
}

void Simulation::initialize() {
    grid.initialize(X_GRID, Y_GRID, Z_GRID);

    // Add collision objects (3D boundaries and obstacle spheres)
    float wallThickness = 2.0f;
    float wall_friction = 0.2f;

    // 6 Domain Boundary Boxes
    collisionManager.addBox(Vector3f(wallThickness * 0.5f, Y_GRID * 0.5f, Z_GRID * 0.5f), Vector3f(wallThickness * 0.5f, Y_GRID * 0.5f, Z_GRID * 0.5f), 0.0f, wall_friction);
    collisionManager.addBox(Vector3f(X_GRID - wallThickness * 0.5f, Y_GRID * 0.5f, Z_GRID * 0.5f), Vector3f(wallThickness * 0.5f, Y_GRID * 0.5f, Z_GRID * 0.5f), 0.0f, wall_friction);
    collisionManager.addBox(Vector3f(X_GRID * 0.5f, wallThickness * 0.5f, Z_GRID * 0.5f), Vector3f(X_GRID * 0.5f, wallThickness * 0.5f, Z_GRID * 0.5f), 0.0f, wall_friction);
    collisionManager.addBox(Vector3f(X_GRID * 0.5f, Y_GRID - wallThickness * 0.5f, Z_GRID * 0.5f), Vector3f(X_GRID * 0.5f, wallThickness * 0.5f, Z_GRID * 0.5f), 0.0f, wall_friction);
    collisionManager.addBox(Vector3f(X_GRID * 0.5f, Y_GRID * 0.5f, wallThickness * 0.5f), Vector3f(X_GRID * 0.5f, Y_GRID * 0.5f, wallThickness * 0.5f), 0.0f, wall_friction);
    collisionManager.addBox(Vector3f(X_GRID * 0.5f, Y_GRID * 0.5f, Z_GRID - wallThickness * 0.5f), Vector3f(X_GRID * 0.5f, Y_GRID * 0.5f, wallThickness * 0.5f), 0.0f, wall_friction);

    // Obstacle Spheres
    //collisionManager.addSphere(Vector3f(10.0f, 15.0f, Z_GRID * 0.5f), 2.0f, 0.3f);
    //collisionManager.addSphere(Vector3f(30.0f, 5.0f, Z_GRID * 0.5f), 4.0f, 0.3f);
    collisionManager.copyToDevice();

    // Spawn an initial shape of material
    std::vector<Vector3f> init_pos;
    std::vector<Vector3f> init_displacement;

    Vector3f init_vel(0.0f, 0.0f, 0.0f);

    ps.water.allocate();
    ps.snow.allocate();
    ps.elastic.allocate();

    if (initSphere) {
        init_pos.reserve(MAX_PARTICLES);
        init_displacement.reserve(MAX_PARTICLES);

        Vector3f center(static_cast<float>(X_GRID) * 0.5f, static_cast<float>(Y_GRID) * 0.5f, static_cast<float>(Z_GRID) * 0.5f);

        float radius = 10.0f; // Size of sphere
        float spacing = CELL_SPACING; // Distance between particles

        std::mt19937 gen(1337);
        float jitterAmount = 0.35f * spacing;
        std::uniform_real_distribution<float> dist(-jitterAmount, jitterAmount);

        // Generate particles in a 3D sphere with jitter
        for (float x = -radius; x <= radius; x += spacing) {
            for (float y = -radius; y <= radius; y += spacing) {
                for (float z = -radius; z <= radius; z += spacing) {
                    if (x * x + y * y + z * z <= radius * radius) {
                        // Apply jitter offset to position
                        float px = center.x + x + dist(gen);
                        float py = center.y + y + dist(gen);
                        float pz = center.z + z + dist(gen);

                        init_pos.push_back(Vector3f(px, py, pz));
                        init_displacement.push_back(physicsDt * init_vel);
                    }
                }
            }
        }

        int particle_count = static_cast<int>(init_pos.size());

        if (materialType == 0)
            ps.water.initialize(particle_count, init_pos, init_displacement);
        else if (materialType == 1)
            ps.snow.initialize(particle_count, init_pos, init_displacement);
        else
            ps.elastic.initialize(particle_count, init_pos, init_displacement);
    }
}

void AddParticlesMidSim(SimulationParticles& ps, float dt, int substepCount, int materialType) {
    if (substepCount <= 0) return;

    static float emission_phase = 0.0f;

    std::vector<Vector3f> init_pos;
    std::vector<Vector3f> init_displacement;

    Vector3f init_vel(100.0f, 0.0f, 0.0f);

    float emit_x = static_cast<float>(INT_CELL_SPAN);
    float emit_y_center = static_cast<float>(Y_GRID) - 4.0f;
    float emit_z_center = static_cast<float>(Z_GRID) * 0.5f;

    float nozzle_radius = 2.0f;
    float spacing = CELL_SPACING;

    // Total distance the fluid mass will move forward during this frame
    float total_dist = init_vel.x * dt * substepCount;

    float current_dist_offset = emission_phase;

    // Emit while the total distance is not satisfied to keep a constant nozzle
    while (current_dist_offset < total_dist) {
        float slice_x = emit_x - current_dist_offset;

        for (float dy = -nozzle_radius; dy <= nozzle_radius; dy += spacing) {
            for (float dz = -nozzle_radius; dz <= nozzle_radius; dz += spacing) {
                if (dy * dy + dz * dz <= nozzle_radius * nozzle_radius) {

                    float jitter_y = ((float)rand() / RAND_MAX - 0.5f) * 0.1f * spacing;
                    float jitter_z = ((float)rand() / RAND_MAX - 0.5f) * 0.1f * spacing;

                    Vector3f pos(
                        slice_x,
                        emit_y_center + dy + jitter_y,
                        emit_z_center + dz + jitter_z
                    );

                    init_pos.push_back(pos);
                    init_displacement.push_back(dt * init_vel);
                }
            }
        }
        current_dist_offset += spacing;
    }

    emission_phase = current_dist_offset - total_dist;

    if (init_pos.empty()) return;

    if (materialType == 0)
        ps.water.addParticlesMidSimulation(init_pos, init_displacement);
    else if (materialType == 1)
        ps.snow.addParticlesMidSimulation(init_pos, init_displacement);
    else
        ps.elastic.addParticlesMidSimulation(init_pos, init_displacement);
}
void Simulation::step(float renderDt) {
    if (isPaused || (ps.getParticlesCount() == 0 && !addMidSim)) return;

    // Time accumulator to step physics stably
    if (renderDt > 0.1f) renderDt = 0.1f;

    timeAccumulator += renderDt;

    int substepCount = static_cast<int>(timeAccumulator / physicsDt);

    const int MAX_SUBSTEPS = 20; // Prevent spiral of death
    if (substepCount > MAX_SUBSTEPS) {
        substepCount = MAX_SUBSTEPS;
    }

    timeAccumulator -= static_cast<float>(substepCount) * physicsDt;

    if (ps.getParticlesCount() < MAX_PARTICLES && addMidSim) {
        AddParticlesMidSim(ps, physicsDt, substepCount, materialType);
    }

    // Execute physics steps
    for (int substep = 0; substep < substepCount; ++substep) {

        // PB-MPM solver iterations
        for (int i = 0; i < solverIterations; i++) {
            if (ps.water.num_particles != 0) solveConstraints(ps.water);
            if (ps.snow.num_particles != 0) solveConstraints(ps.snow);
            if (ps.elastic.num_particles != 0) solveConstraints(ps.elastic);

            grid.clear();

            if (ps.water.num_particles != 0) p2g(ps.water, grid);
            if (ps.snow.num_particles != 0) p2g(ps.snow, grid);
            if (ps.elastic.num_particles != 0) p2g(ps.elastic, grid);

            updateGrid(grid, collisionManager.getDeviceData());
            if (ps.water.num_particles != 0) g2p(ps.water, grid);
            if (ps.snow.num_particles != 0) g2p(ps.snow, grid);
            if (ps.elastic.num_particles != 0) g2p(ps.elastic, grid);
        }

        if (ps.water.num_particles != 0) integrateParticle(ps.water, grid, physicsDt, gravity, collisionManager.getDeviceData());
        if (ps.snow.num_particles != 0) integrateParticle(ps.snow, grid, physicsDt, gravity, collisionManager.getDeviceData());
        if (ps.elastic.num_particles != 0) integrateParticle(ps.elastic, grid, physicsDt, gravity, collisionManager.getDeviceData());
    }
}

void Simulation::updateMaterialSettings(MaterialType type, MaterialSettings settings) {
    switch (type) {
    case MaterialType::WATER:
        ps.water.settings = settings;
        break;
    case MaterialType::SNOW:
        ps.snow.settings = settings;
        break;
    case MaterialType::ELASTIC:
        ps.elastic.settings = settings;
        break;
    }
}

void Simulation::free() {
    ps.free();
    grid.free();
}