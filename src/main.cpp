
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <iostream>
#include <vector>
#include "constants.h"
#include "solver.cuh"
#include "grid.h"
#include "particleSystem.h"
#include "boundary.h"
#include "types.h"

#ifdef _WIN32
extern "C" {
    __declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
}
#endif

/* Globals */
Grid grid;
ParticleSystem ps;
BoundaryData bounds;

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

#pragma region Material Point Method Algorithm
void Initialization()
{
    grid.initialize(X_GRID, Y_GRID);

    BoundaryManager boundaryManager;
    boundaryManager.InitializeDefaultBorders();
    boundaryManager.CopyToDevice();
    bounds = boundaryManager.GetDeviceData();

    // Spawn fluid block (e.g., 40x40 block of particles)
    std::vector<Vector2f> init_pos;
    std::vector<Vector2f> init_vel;

    ps.initialize(static_cast<int>(init_pos.size()), init_pos, init_vel);
}

void AddParticles() {
    std::vector<Vector2f> new_pos;
    std::vector<Vector2f> new_vel;

    Vector2f v(30.0f, 0.0f);

    for (int p = 0; p < 8; ++p) {
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

        float pos_x = static_cast<float>(INT_CELL_SPAN);
        float pos_y = static_cast<float>(Y_GRID) - 2.0f * static_cast<float>(INT_CELL_SPAN) - 0.5f * static_cast<float>(p) - r;

        new_pos.push_back(make_float2(pos_x, pos_y));
        new_vel.push_back(v);
    }

    ps.addParticlesMidSimulation(new_pos, new_vel, 1.14f, 0.0005f);
}

void Update()
{
    if (ps.num_particles < MAX_PARTICLES && stepCount % EMISSION_INTERVAL == 0) {
        AddParticles();
    }

    stepCount++;

    grid.clear();
    p2g(ps, grid, PHYSICS_DT);
    updateGrid(grid, PHYSICS_DT, bounds);
    g2p(ps, grid, PHYSICS_DT);
}
#pragma endregion

#pragma region OpenGL

void InitOpenGLInterop()
{
    // 1. Generate OpenGL buffer to store particles
    glGenBuffers(1, &particleVBO);
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);

    // 2. Allocate enough memory for max particles
    glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLES * sizeof(Vector2f), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // 3. Register the buffer with CUDA
    cudaGraphicsGLRegisterBuffer(&particleCudaResource, particleVBO, cudaGraphicsMapFlagsWriteDiscard);
}

void RenderParticles()
{
    if (ps.num_particles == 0) return;

    // 1. Map the OpenGL resource to CUDA ptr
    cudaGraphicsMapResources(1, &particleCudaResource, 0);

    Vector2f* d_vbo_ptr;
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_ptr, &num_bytes, particleCudaResource);

    // 2. Copy directly from device to device
    cudaMemcpy(d_vbo_ptr, ps.d_Xp, ps.num_particles * sizeof(Vector2f), cudaMemcpyDeviceToDevice);

    // 3. Unmap the resource so OpenGL can use it again
    cudaGraphicsUnmapResources(1, &particleCudaResource, 0);

    // 4. Render
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    glEnableClientState(GL_VERTEX_ARRAY);

    glVertexPointer(2, GL_FLOAT, 0, (void*)0);

    glColor3f(0.2f, 0.6f, 1.0f);
    glEnable(GL_POINT_SMOOTH);
    glPointSize(12);

    glDrawArrays(GL_POINTS, 0, ps.num_particles);

    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void RenderGridNodes() {
    std::vector<float> h_Mi(grid.num_nodes);
    cudaMemcpy(h_Mi.data(), grid.d_Mi, (grid.num_nodes) * sizeof(float), cudaMemcpyDeviceToHost);

    glEnable(GL_POINT_SMOOTH);
    glPointSize(3.0f);

    glBegin(GL_POINTS);
    int stride = grid.grid_x + 1;

    for (int i = 0; i < grid.num_nodes; ++i) {
        // Derive position on the fly on CPU
        float gx = static_cast<float>(i % stride);
        float gy = static_cast<float>(i / stride);

        if (h_Mi[i] > 0.0f) {
            glColor3f(0.5f, 0.5f, 0.5f); // Active node (has mass)
        }
        else {
            glColor3f(0.3f, 0.3f, 0.3f); // Inactive node
        }

        glVertex2f(gx, gy);
    }

    glEnd();
}

GLFWwindow* initGLFWContext()
{
    if (!glfwInit()) exit(EXIT_FAILURE);

    GLFWwindow* window = glfwCreateWindow(X_WINDOW, Y_WINDOW, "CUDA MPM Simulation", NULL, NULL);
    if (!window) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window);

    GLenum err = glewInit();
    if (GLEW_OK != err) {
        std::cerr << "[CRASH] Error initializing GLEW: " << glewGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

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

void Free_OpenGL() {
    cudaGraphicsUnregisterResource(particleCudaResource);
    glDeleteBuffers(1, &particleVBO);
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

        InitOpenGLInterop();

        while (!glfwWindowShouldClose(window))
        {
            glClear(GL_COLOR_BUFFER_BIT);

            // window renders every frame but simulation runs (in substeps) only when unpased
            if (!pauseSimulation || stepOnce) {

                for(int step = 0; step < SIM_SUBSTEPS; step++)
                    Update();

                stepOnce = false;
            }

            //RenderGridNodes();
            RenderParticles();

            glfwSwapBuffers(window);
            glfwPollEvents();
        }

        ps.free();
        grid.free();
        glfwTerminate();
        Free_OpenGL();
    }
    catch (const std::exception& e) {
        std::cerr << "[CRASH] Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
#pragma endregion