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
    float dt;
    bool isPaused;

    Simulation();
    ~Simulation();

    void initialize();
    void step(); 
    void free();
    

    // Getter methods for the renderer
    const SimulationParticles& getParticles() const { return ps; }
    const CollisionManager& getCollisionManager() const { return collisionManager; }
};