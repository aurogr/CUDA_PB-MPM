#pragma once

/* ----- GRID ----- */
inline constexpr int X_GRID = 50; // Size of the domain
inline constexpr int Y_GRID = 25;
inline constexpr int Z_GRID = 25;

inline constexpr float CELL_SPACING = 0.5f; // In 3D thats 8 particles per cell

inline constexpr float COMPUTED_VP0 = CELL_SPACING * CELL_SPACING * CELL_SPACING;

inline constexpr float COMPUTED_MP0 = 1.0f * COMPUTED_VP0; // density set to 1.0

// for simplicity we are gonna stablish that the cell size is 1.0 so that we can omit it from the code
inline constexpr float H = 1.0f;

/* ----- RENDERING ----- */
inline constexpr int X_WINDOW = 1080; // Window size
inline constexpr int Y_WINDOW = X_WINDOW * Y_GRID / X_GRID;

/* ----- SIMULATION ----- */
inline constexpr int MAX_PARTICLES = 300000;

/* ----- QUADRATIC INTERPOLATION ----- */
inline constexpr float INT_CELL_SPAN = 1.5f;
