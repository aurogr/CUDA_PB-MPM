#pragma once

/* ----- GRID ----- */
const static int X_GRID = 200; // Size of the domain
const static int Y_GRID = 100;

// for simplicity we are gonna stablish that the cell size is 1.0 so that we can omit it from the code
//const static double H = 1.0; 
//const static double H_INV = 1.0;

/* ----- RENDERING ----- */
const static int X_WINDOW = 1400; // Window size
const static int Y_WINDOW = X_WINDOW * Y_GRID / X_GRID;

const static double RENDER_DT = 0.016; // 60 FPS

/* ----- SIMULATION ----- */
inline constexpr int MAX_PARTICLES = 5000;

const static double PHYSICS_DT = 0.0001;	// Physics timeStep needed for stability (for water it cannot be higher than 0.001)

const static int SIM_SUBSTEPS = static_cast<int>(RENDER_DT / PHYSICS_DT); // Simulation substeps needed to control the render framerate

const static int EMISSION_INTERVAL = SIM_SUBSTEPS; // Rate of particles addition (if = to SIM_SUBSTEPS it emits particles every rendered frame)

const bool INIT_SPHERE = false;
const bool ADD_MID_SIM = true;

/* ----- QUADRATIC INTERPOLATION ----- */
const static float INT_CELL_SPAN = 1.5f;
