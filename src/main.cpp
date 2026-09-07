
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <iostream>
#include <vector>
#include "constants.h"
#include "solver.cuh"
#include "types.h"
#include "render.h"

/* Globals */
Simulation simEngine;
GLRenderer renderEngine;

int stepCount = 0;

void initGLContext();
GLFWwindow* initGLFWContext();

bool pauseSimulation = true;
bool stepOnce = false;

GLuint particleVBO;
cudaGraphicsResource_t particleCudaResource;

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

#pragma region Main
int main()
{
    std::cout << "[INFO] Starting PB-MPM Simulation..." << std::endl;

    if (!glfwInit()) {
        std::cerr << "[ERROR] Failed to initialize GLFW" << std::endl;
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(X_WINDOW, Y_WINDOW, "C++/CUDA PB-MPM SIMULATION", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    // Instantiate decoupled physics and rendering engines
    Simulation sim;
    GLRenderer renderer;

    sim.initialize();
    renderer.initializeGL();
    renderer.resizeViewport(X_WINDOW, Y_WINDOW);

    // Main render loop
    while (!glfwWindowShouldClose(window))
    {
        sim.step();
        renderer.render(sim);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    sim.free();
    renderer.freeGL();
    glfwTerminate();

    return 0;
}
#pragma endregion