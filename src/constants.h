#pragma once

/* ----- GRID ----- */
const static int X_GRID = 100; // Size of the domain
const static int Y_GRID = 50;

const static int PARTICLES_PER_CELL_AXIS = 2;

const static float CELL_SPACING = 1.0 / PARTICLES_PER_CELL_AXIS;

const static float COMPUTED_VP0 = CELL_SPACING * CELL_SPACING;

const static float COMPUTED_MP0 = 1.0 * COMPUTED_VP0; // density set to 1.0

// for simplicity we are gonna stablish that the cell size is 1.0 so that we can omit it from the code
const static double H = 1.0; 
//const static double H_INV = 1.0;

/* ----- RENDERING ----- */
const static int X_WINDOW = 1080; // Window size
const static int Y_WINDOW = X_WINDOW * Y_GRID / X_GRID;

const static double RENDER_DT = 0.016; // 60 FPS

/* ----- SIMULATION ----- */
inline constexpr int MAX_PARTICLES = 15000;

const static float PHYSICS_DT = 0.05f;

const static float GRAVITY = 9.81f;

const static int SIM_SUBSTEPS = static_cast<int>(RENDER_DT / PHYSICS_DT); // Simulation substeps needed to control the render framerate

const static int EMISSION_INTERVAL = SIM_SUBSTEPS; // Rate of particles addition (if = to SIM_SUBSTEPS it emits particles every rendered frame)

const bool INIT_SPHERE = true;
const bool ADD_MID_SIM = false;
const int SIM_STEPS = 10;

/* ----- QUADRATIC INTERPOLATION ----- */
const static float INT_CELL_SPAN = 1.5f;
