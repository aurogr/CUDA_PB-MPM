
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

/* Globals */
Grid grid;
ParticleSystem<WaterData> ps;
LevelSetCollisionManager collisionManager;

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

    // Add collision objects
    float wallThickness = 2.0f; // Thickness of the solid wall
    float wall_friction = 0.2f;

    // Left border
    collisionManager.addBox(
        Vector2f(wallThickness * 0.5f, Y_GRID * 0.5f),
        Vector2f(wallThickness * 0.5f, Y_GRID * 0.5f),
        0.0f, wall_friction
    );

    // Right Border 
    collisionManager.addBox(
        Vector2f(X_GRID - wallThickness * 0.5f, Y_GRID * 0.5f),
        Vector2f(wallThickness * 0.5f, Y_GRID * 0.5f),
        0.0f, wall_friction
    );

    // Bottom Border
    collisionManager.addBox(
        Vector2f(X_GRID * 0.5f, wallThickness * 0.5f),
        Vector2f(X_GRID * 0.5f, wallThickness * 0.5f),
        0.0f, wall_friction
    );

    // Top Border
    collisionManager.addBox(
        Vector2f(X_GRID * 0.5f, Y_GRID - wallThickness * 0.5f),
        Vector2f(X_GRID * 0.5f, wallThickness * 0.5f),
        0.0f, wall_friction
    );

    // Collider sphere
    collisionManager.addSphere(Vector2f(20.0f, 15.0f), 10.0f, .3f);

    collisionManager.copyToDevice();

    // Spawn an initial shape of fluid
    std::vector<Vector2f> init_pos;
    std::vector<Vector2f> init_displacement;

    Vector2f init_vel(0.0f, 0.0f);

    if (INIT_SPHERE) {
        Vector2f center(static_cast<float>(X_GRID) * 0.5f, static_cast<float>(Y_GRID) * 0.5f);

        float radius = 20.0f; // Size of sphere
        float spacing = CELL_SPACING; // Distance between particles

        // Generate particles in a circle
        for (float x = -radius; x <= radius; x += spacing) {
            for (float y = -radius; y <= radius; y += spacing) {
                if (x * x + y * y <= radius * radius) {
                    init_pos.push_back(Vector2f(center.x + x, center.y + y));
                    init_displacement.push_back(PHYSICS_DT * init_vel);
                }
            }
        }

        int particle_count = static_cast<int>(init_pos.size());

        // Pass the corrected values into your particle system initialization
        ps.initialize(particle_count, init_pos, init_displacement);
    } else
        ps.initialize(static_cast<int>(init_pos.size()), init_pos, init_displacement);
}

void AddParticles() {
    std::vector<Vector2f> init_pos;
    std::vector<Vector2f> init_displacement;

    Vector2f init_vel(10.0f, 0.0f);

    for (int p = 0; p < 8; ++p) {
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        init_pos.push_back(Vector2f(static_cast<float>(INT_CELL_SPAN), static_cast<float>(Y_GRID) - 2.0f * static_cast<float>(INT_CELL_SPAN) - 0.5f * static_cast<float>(p) - r));
        init_displacement.push_back(PHYSICS_DT * init_vel);
    }

    ps.addParticlesMidSimulation(init_pos, init_displacement);
}

void Update()
{
    if (ps.num_particles < MAX_PARTICLES && stepCount % EMISSION_INTERVAL == 0 && ADD_MID_SIM) {
        AddParticles();
    }
     
    stepCount++;

    for (int i = 0; i < SIM_STEPS; i++) {
        solveConstraints(ps);
        grid.clear();
        p2g(ps, grid);
        updateGrid(grid, collisionManager.getDeviceData());
        g2p(ps, grid);
    }
    integrateParticle(ps, grid, PHYSICS_DT, collisionManager.getDeviceData());
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
    glPointSize(3.0);

    glDrawArrays(GL_POINTS, 0, ps.num_particles);

    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void RenderGridBackground()
{
    // Save current OpenGL states if needed, or set up color for grid lines (e.g., subtle gray)
    glColor3f(0.2f, 0.2f, 0.2f); // Dark gray color for grid lines
    glLineWidth(1.0f);

    glBegin(GL_LINES);

    // Draw vertical grid lines
    for (int x = 0; x <= X_GRID; ++x)
    {
        float xPos = static_cast<float>(x) * H;
        glVertex2f(xPos, 0.0f);
        glVertex2f(xPos, static_cast<float>(Y_GRID) * H);
    }

    // Draw horizontal grid lines
    for (int y = 0; y <= Y_GRID; ++y)
    {
        float yPos = static_cast<float>(y) * H;
        glVertex2f(0.0f, yPos);
        glVertex2f(static_cast<float>(X_GRID) * H, yPos);
    }

    glEnd();
}

void RenderColliders() {
    glColor3f(.5f, .0f, .0f);

    for (const auto& obj : collisionManager.h_objects) {
        for (const auto& obj : collisionManager.h_objects) {
            if (obj.type == 1) {
                float cx = obj.center.x;
                float cy = obj.center.y;
                float hx = obj.size.x; // Half-width
                float hy = obj.size.y; // Half-height

                if (obj.rotation == 0.0f) {
                    glBegin(GL_QUADS);
                    glVertex2f(cx - hx, cy - hy); // Bottom-left
                    glVertex2f(cx + hx, cy - hy); // Bottom-right
                    glVertex2f(cx + hx, cy + hy); // Top-right
                    glVertex2f(cx - hx, cy + hy); // Top-left
                    glEnd();
                }
                else {
                    float c = cosf(obj.rotation);
                    float s = sinf(obj.rotation);

                    auto rotatePoint = [cx, cy, c, s](float localX, float localY) {
                        float rx = c * localX - s * localY;
                        float ry = s * localX + c * localY;
                        return Vector2f(cx + rx, cy + ry);
                        };

                    Vector2f bl = rotatePoint(-hx, -hy);
                    Vector2f br = rotatePoint(hx, -hy);
                    Vector2f tr = rotatePoint(hx, hy);
                    Vector2f tl = rotatePoint(-hx, hy);

                    glBegin(GL_QUADS);
                    glVertex2f(bl.x, bl.y);
                    glVertex2f(br.x, br.y);
                    glVertex2f(tr.x, tr.y);
                    glVertex2f(tl.x, tl.y);
                    glEnd();
                }
            }
            else if (obj.type == 0) {
                float cx = obj.center.x;
                float cy = obj.center.y;
                float radius = obj.size.x; // Assuming radius is stored in size.x
                int segments = 20;         // Smoothness of the circle

                glBegin(GL_TRIANGLE_FAN);
                glVertex2f(cx, cy); // Center of the circle for the fan

                for (int i = 0; i <= segments; ++i) {
                    float theta = 2.0f * 3.1415926f * static_cast<float>(i) / static_cast<float>(segments);
                    float x = cx + radius * cosf(theta);
                    float y = cy + radius * sinf(theta);
                    glVertex2f(x, y);
                }
                glEnd();
            }
        }
    }
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

                Update();

                stepOnce = false;
            }

            RenderGridBackground();
            RenderColliders();
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