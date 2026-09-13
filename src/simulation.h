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
    float dt = 0.001f;
    bool isPaused = false;
    bool initSphere = true;
    bool addMidSim = true;
    const float gravity = 9.81f;
    const int solverIterations = 4;

    int materialType = 0; // 0 = water, 1 = snow, 2 = elastic, 3 = both TODO: CHANGE FOR PREPARED SCENES

    Simulation();
    ~Simulation();

    void initialize();
    void step(); 
    void free();
    void togglePause();
    void updateMaterialSettings(MaterialType type, MaterialSettings settings);

    // Getter methods for the renderer
    const SimulationParticles& getParticles() const { return ps; }
    const CollisionManager& getCollisionManager() const { return collisionManager; }
};