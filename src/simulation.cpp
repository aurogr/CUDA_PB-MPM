#include <cuda_runtime.h>

#include "simulation.h"
#include "constants.h"
#include "solver.cuh"

Simulation::Simulation()
    : dt(PHYSICS_DT), isPaused(false) {
}

Simulation::~Simulation() {
    free();
}

void Simulation::togglePause() { isPaused = !isPaused; }

void Simulation::initialize() {
    grid.initialize(X_GRID, Y_GRID);

    // Add collision objects
    float wallThickness = 2.0f;
    float wall_friction = 0.2f;

    collisionManager.addBox(Vector2f(wallThickness * 0.5f, Y_GRID * 0.5f), Vector2f(wallThickness * 0.5f, Y_GRID * 0.5f), 0.0f, wall_friction);
    collisionManager.addBox(Vector2f(X_GRID - wallThickness * 0.5f, Y_GRID * 0.5f), Vector2f(wallThickness * 0.5f, Y_GRID * 0.5f), 0.0f, wall_friction);
    collisionManager.addBox(Vector2f(X_GRID * 0.5f, wallThickness * 0.5f), Vector2f(X_GRID * 0.5f, wallThickness * 0.5f), 0.0f, wall_friction);
    collisionManager.addBox(Vector2f(X_GRID * 0.5f, Y_GRID - wallThickness * 0.5f), Vector2f(X_GRID * 0.5f, wallThickness * 0.5f), 0.0f, wall_friction);
    collisionManager.addSphere(Vector2f(20.0f, 15.0f), 10.0f, .3f);
    collisionManager.copyToDevice();

    // Spawn an initial shape of water
    std::vector<Vector2f> init_pos;
    std::vector<Vector2f> init_displacement;

    Vector2f init_vel(0.0f, 0.0f);

    if (INIT_SPHERE) {

        init_pos.reserve(MAX_PARTICLES);
        init_displacement.reserve(MAX_PARTICLES);

        Vector2f center(static_cast<float>(X_GRID) * 0.5f, static_cast<float>(Y_GRID) * 0.5f);

        float radius = 20.0f; // Size of sphere
        float spacing = CELL_SPACING; // Distance between particles

        // Generate particles in a circle
        for (float x = -radius; x <= radius; x += spacing) {
            for (float y = -radius; y <= radius; y += spacing) {
                if (x * x + y * y <= radius * radius) {
                    init_pos.push_back(Vector2f(center.x + x, center.y + y));
                    init_displacement.push_back(PHYSICS_DT * init_vel);
                }
            }
        }

        int particle_count = static_cast<int>(init_pos.size());

        // Pass the corrected values into your particle system initialization
        ps.water.initialize(particle_count, init_pos, init_displacement);
    }
    else
        ps.water.initialize(static_cast<int>(init_pos.size()), init_pos, init_displacement);
}

void AddParticlesMidSim(SimulationParticles& ps) {
    // Add water particles mid sim
    std::vector<Vector2f> init_pos;
    std::vector<Vector2f> init_displacement;

    Vector2f init_vel(10.0f, 0.0f);

    for (int p = 0; p < 8; ++p) {
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        init_pos.push_back(Vector2f(static_cast<float>(INT_CELL_SPAN), static_cast<float>(Y_GRID) - 2.0f * static_cast<float>(INT_CELL_SPAN) - 0.5f * static_cast<float>(p) - r));
        init_displacement.push_back(PHYSICS_DT * init_vel);
    }

    ps.water.addParticlesMidSimulation(init_pos, init_displacement);
}

void Simulation::step() {
    if (isPaused || ps.getParticlesCount() == 0) return;

    stepCount++;

    if (ps.getParticlesCount() < MAX_PARTICLES && stepCount % EMISSION_INTERVAL == 0 && ADD_MID_SIM) {
        AddParticlesMidSim(ps);
    }

    for (int i = 0; i < SOLVER_ITERATIONS; i++) {

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

    if (ps.water.num_particles != 0) integrateParticle(ps.water, grid, dt, collisionManager.getDeviceData());
    if (ps.snow.num_particles != 0) integrateParticle(ps.snow, grid, dt, collisionManager.getDeviceData());
    if (ps.elastic.num_particles != 0) integrateParticle(ps.elastic, grid, dt, collisionManager.getDeviceData());
}

void Simulation::free() {
    ps.free();
    grid.free();
}