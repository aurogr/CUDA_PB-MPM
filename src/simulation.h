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
    float timeAccumulator = 0.0f;

    // Variables passed from interface
    float physicsDt = 0.001f;
    bool isPaused = false;
    bool initSphere = true;
    bool addMidSim = true;
    int materialType = 2; // 0 = water, 1 = snow, 2 = elastic, 3 = both TODO: CHANGE FOR PREPARED SCENES

    // Constant variables
    const float gravity = 9.81f;
    const int solverIterations = 4;

public:
    Simulation();
    ~Simulation();

    void initialize();
    void step(float renderDt); 
    void free();
    void updateMaterialSettings(MaterialType type, MaterialSettings settings);

#pragma region Getters
    const SimulationParticles& getParticles() const { return ps; }
    const CollisionManager& getCollisionManager() const { return collisionManager; }
#pragma endregion

#pragma region Setters
    void setPause(float newValue) { isPaused = newValue; }
    void setPhysicsDt(float newValue) { physicsDt = newValue; }
    void setInitSphere(float newValue) { initSphere = newValue; }
    void setAddMidSim(float newValue) { addMidSim = newValue; }
    void setMaterialType(float newValue) { materialType = newValue; }
#pragma endregion
};