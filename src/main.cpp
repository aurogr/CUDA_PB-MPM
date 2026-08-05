#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "constants.h"
#include "solver.cuh"
#include "grid.h"
#include "particleSystem.h"
#include "boundary.h"

#include <GLFW/glfw3.h>

/* Globals */
Grid grid;
ParticleSystem ps;
BoundaryData bounds;

int frameCount = 0;

void initGLContext();
GLFWwindow* initGLFWContext();

bool pauseSimulation = true;
bool stepOnce = false;

// Key callback function
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_SPACE) {
            pauseSimulation = !pauseSimulation; // Toggle play/pause
        }
        if (key == GLFW_KEY_S) {
            stepOnce = true; // Step exactly 1 frame forward
        }
    }
}

#pragma region Material Point Method Algorithm
void Initialization()
{
    grid.initialize(X_GRID, Y_GRID);

    BoundaryManager boundaryManager;
    boundaryManager.InitializeDefaultBorders();
    boundaryManager.CopyToDevice();
    bounds = boundaryManager.GetDeviceData();

    // Spawn fluid block (e.g., 40x40 block of particles)
    std::vector<float2> init_pos;
    std::vector<float2> init_vel;

    ps.initialize(static_cast<int>(init_pos.size()), init_pos, init_vel);
}

void AddParticles() {
    std::vector<float2> new_pos;
    std::vector<float2> new_vel;

    float2 v = make_float2(30.0f, 0.0f); // Initial velocity[cite: 10]

    for (int p = 0; p < 8; ++p) { //[cite: 10]
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); //[cite: 10]

        // Formula translated directly from your snippet[cite: 10]:
        float pos_x = static_cast<float>(CUB); //[cite: 10]
        float pos_y = static_cast<float>(Y_GRID) - 2.0f * static_cast<float>(CUB) - 0.5f * static_cast<float>(p) - r; //[cite: 10]

        new_pos.push_back(make_float2(pos_x, pos_y));
        new_vel.push_back(v);
    }

    // Append to GPU particle system (Water parameters: Vp0 = 1.14, Mp = 0.0005)[cite: 10]
    ps.addParticlesMidSimulation(new_pos, new_vel, 1.14f, 0.0005f);
}

void Update()
{
    if (frameCount % DT_ROB == 0) {
        AddParticles();
    }
    frameCount++;

    grid.clear();
    p2g(ps, grid, DT);
    updateGrid(grid, DT, bounds);
    g2p(ps, grid, DT);
}

void RenderParticles()
{
    if (ps.num_particles == 0) return;

    std::vector<float2> h_Xp(ps.num_particles);
    cudaMemcpy(h_Xp.data(), ps.d_Xp, ps.num_particles * sizeof(float2), cudaMemcpyDeviceToHost);

    glColor3f(0.2f, 0.6f, 1.0f);
    glPointSize(12);

    glEnable(GL_POINT_SMOOTH);
    glBegin(GL_POINTS);
    for (int i = 0; i < ps.num_particles; ++i) {
        glVertex2f(h_Xp[i].x, h_Xp[i].y);
    }
    glEnd();
}
#pragma endregion

#pragma region Main
int main()
{
    std::cout << "[INFO] Starting CUDA MPM Simulation..." << std::endl;

    try {
        Initialization();

        GLFWwindow* window = initGLFWContext();
        if (!window) return -1;

        glfwSetKeyCallback(window, key_callback);

        initGLContext();

        while (!glfwWindowShouldClose(window))
        {
            glClear(GL_COLOR_BUFFER_BIT);

            // Only update if not paused, or if single-step was requested
            if (!pauseSimulation || stepOnce) {
                Update();
                stepOnce = false;
            }

            RenderParticles();

            glfwSwapBuffers(window);
            glfwPollEvents();
        }

        ps.free();
        grid.free();
        glfwTerminate();
    }
    catch (const std::exception& e) {
        std::cerr << "[CRASH] Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
#pragma endregion

#pragma region OpenGL
GLFWwindow* initGLFWContext()
{
    if (!glfwInit()) exit(EXIT_FAILURE);

    GLFWwindow* window = glfwCreateWindow(X_WINDOW, Y_WINDOW, "CUDA MPM Simulation", NULL, NULL);
    if (!window) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window);
    return window;
}

void initGLContext()
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, X_GRID, 0, Y_GRID, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, (GLsizei)X_WINDOW, (GLsizei)Y_WINDOW);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
}
#pragma endregion
