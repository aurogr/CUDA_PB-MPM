#pragma once

#include <iostream>
#include "grid.h"
#include "particleSystem.h"
#include "boundary.h"
#include "types.h"
#include <list>

class Simulation {
private:
    Grid grid;
    SimulationParticles ps;
    CollisionManager collisionManager;
    int stepCount = 0;

public:
    float dt = 0.05f;
    bool isPaused = true;
    bool init_sphere = true;
    bool add_mid_sim = false;
    const float gravity = 9.81f;
    const int solver_iterations = 4;

    //const static int SIM_SUBSTEPS = std::max(1, static_cast<int>(RENDER_DT / PHYSICS_DT)); // Simulation substeps needed to control the render framerate
    //const static int EMISSION_INTERVAL = SIM_SUBSTEPS; // Rate of particles addition (if = to SIM_SUBSTEPS it emits particles every rendered frame)

    Simulation();
    ~Simulation();

    void initialize();
    void step(); 
    void free();
    void togglePause();

    // Getter methods for the renderer
    const SimulationParticles& getParticles() const { return ps; }
    const CollisionManager& getCollisionManager() const { return collisionManager; }
};