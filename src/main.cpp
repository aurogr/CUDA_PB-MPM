
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <chrono>
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
                    simEngine.setPhysicsDt(newDt);
                }
            }
            else if (command == "PAUSE") {
                int pauseState;
                if (ss >> pauseState) {
                    simEngine.setPause(pauseState);
                }
            }
            else if (command == "MID_SIM") {
                int add_mid_sim;
                if (ss >> add_mid_sim) {
                    simEngine.setAddMidSim(add_mid_sim);
                }
            }
            else if (command == "SET_MAT_SETTINGS") {
                int matType;
                MaterialSettings s;
                if (ss >> matType >> s.relaxation >> s.viscosity
                    >> s.crit_compression >> s.crit_stretch >> s.hard_coeff >> s.elasticity_ratio)
                {
                    simEngine.updateMaterialSettings((MaterialType)matType, s);
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
    int winX = 100;
    int winY = 500;

    // Get arguments from python launcher (if there are none we aren't using python)
    if (argc > 1) {
        winX = std::stoi(argv[1]);
        winY = std::stoi(argv[2]);
        simEngine.setInitSphere((std::atoi(argv[3]) == 1));
        simEngine.setAddMidSim((std::atoi(argv[4]) == 1));
        simEngine.setPhysicsDt(static_cast<float>(std::atof(argv[5])));
        simEngine.setPause((std::atoi(argv[6]) == 1));
        simEngine.setMaterialType(std::stoi(argv[7]));

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

    // Instantiate engines
    simEngine.initialize();
    renderEngine.initializeGL();
    renderEngine.resizeViewport(X_WINDOW, Y_WINDOW);

    auto prev_time = std::chrono::high_resolution_clock::now();

    // Main render loop
    while (!glfwWindowShouldClose(window)) { // or your equivalent loop
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = current_time - prev_time;
        prev_time = current_time;

        float actual_render_dt = elapsed.count();

        simEngine.step(actual_render_dt);

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