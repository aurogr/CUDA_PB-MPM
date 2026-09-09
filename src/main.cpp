
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

#pragma region OpenGL RT interaction (deprecated)
// Key callback function
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_SPACE) {
            simEngine.togglePause();
        }
    }
}
#pragma endregion

#pragma region Python RT interaction
void listenToPythonCommands() {
    std::thread([]() {
        std::cout << "[C++ Engine] Command listener thread started." << std::endl;

        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue; // Ignore empty lines

            std::stringstream ss(line);
            std::string command;
            ss >> command;

            if (command == "DT") {
                float newDt;
                if (ss >> newDt) {
                    simEngine.dt = newDt;
                    std::cout << "[C++ Engine] Updated dt to: " << newDt << std::endl;
                }
            }
            else if (command == "PAUSE") {
                int pauseState;
                if (ss >> pauseState) {
                    simEngine.isPaused = pauseState;
                    std::cout << "[C++ Engine] Pause state set to: " << pauseState << std::endl;
                }
            }
        }
        std::cout << "[C++ Engine] Stdin stream closed." << std::endl;
        }).detach();
}
#pragma endregion

#pragma region Main
int main(int argc, char* argv[])
{
    // Default values if run directly from Visual Studio
    float timeStep = 0.05f;
    bool startPaused = false;
    int winX = 100;
    int winY = 500;

    // Get arguments from python launcher (if there are none we aren't using python)
    if (argc > 1) {
        timeStep = static_cast<float>(std::atof(argv[1]));
        startPaused = (std::atoi(argv[2]) == 1); 
        winX = std::stoi(argv[3]);
        winY = std::stoi(argv[4]);

        listenToPythonCommands();
    }    

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

    glfwSetWindowPos(window, winX, winY);

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