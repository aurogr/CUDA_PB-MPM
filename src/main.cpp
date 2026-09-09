
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <iostream>
#include <string> 
#include <sstream> 
#include <thread>    
#include <vector>
#include "constants.h"
#include "solver.cuh"
#include "types.h"
#include "render.h"

/* Globals */
Simulation simEngine;
GLRenderer renderEngine;

// Key callback function
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_SPACE) {
            simEngine.togglePause();
        }
    }
}

void listenToPythonCommands() {
    std::thread([]() {
        std::string line;
        while (std::getline(std::cin, line)) {
            std::stringstream ss(line);
            std::string command;
            ss >> command;

            if (command == "DT") {
                float newDt;
                if (ss >> newDt) {
                    simEngine.dt = newDt; // Modifies dt in real-time[cite: 3]
                }
            }
            else if (command == "PAUSE") {
                int pauseState;
                if (ss >> pauseState) {
                    // Update state if different
                    if ((pauseState == 1) != simEngine.isPaused) {
                        simEngine.togglePause(); // Toggles pause state live[cite: 3]
                    }
                }
            }
        }
        }).detach(); // Detach thread so it runs independently
}

#pragma region Main
int main(int argc, char* argv[])
{
    // Default values if run directly from Visual Studio
    float timeStep = 0.05f;
    bool startPaused = false;

    // Parse command line arguments from Python launcher
    if (argc > 1) timeStep = static_cast<float>(std::atof(argv[1]));
    if (argc > 2) startPaused = (std::atoi(argv[2]) == 1);

    listenToPythonCommands();

    // Create GL window and callbacks
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

    simEngine.dt = timeStep;
    simEngine.isPaused = startPaused;

    // Instantiate engines
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