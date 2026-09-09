
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
            simEngine.togglePause();
        }
        if (key == GLFW_KEY_S) {
            stepOnce = true;
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

    glfwSetKeyCallback(window, key_callback);

    glfwMakeContextCurrent(window);

    // Instantiate decoupled physics and rendering engines

    simEngine.initialize();
    renderEngine.initializeGL();
    renderEngine.resizeViewport(X_WINDOW, Y_WINDOW);

    // Main render loop
    while (!glfwWindowShouldClose(window))
    {
        simEngine.step();
        renderEngine.render(simEngine);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    simEngine.free();
    renderEngine.freeGL();
    glfwTerminate();

    return 0;
}
#pragma endregion