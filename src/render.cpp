#include "render.h"
#include "constants.h"
#include "types.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

// Required for CUDA/OpenGL interop functions
#include <cuda_runtime.h>
#include <cuda_gl_interop.h> 

#include <algorithm>
#include <cmath>

// --- Custom GLU Replacements to avoid library linker errors ---
inline void customPerspective(float fovy, float aspect, float zNear, float zFar) {
    float fH = tanf(fovy / 360.0f * 3.14159265f) * zNear;
    float fW = fH * aspect;
    glFrustum(-fW, fW, -fH, fH, zNear, zFar);
}

inline void customLookAt(float ex, float ey, float ez, float cx, float cy, float cz, float ux, float uy, float uz) {
    float fx = cx - ex, fy = cy - ey, fz = cz - ez;
    float lenF = sqrtf(fx * fx + fy * fy + fz * fz);
    fx /= lenF; fy /= lenF; fz /= lenF;

    float sx = fy * uz - fz * uy, sy = fz * ux - fx * uz, sz = fx * uy - fy * ux;
    float lenS = sqrtf(sx * sx + sy * sy + sz * sz);
    sx /= lenS; sy /= lenS; sz /= lenS;

    float rx = sy * fz - sz * fy, ry = sz * fx - sx * fz, rz = sx * fy - sy * fx;

    float m[16] = {
        sx, rx, -fx, 0.0f,
        sy, ry, -fy, 0.0f,
        sz, rz, -fz, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    glMultMatrixf(m);
    glTranslatef(-ex, -ey, -ez);
}
// ---------------------------------------------------------------

GLRenderer::GLRenderer() {}

GLRenderer::~GLRenderer() {
    freeGL();
}

void GLRenderer::initializeGL() {
    glewInit();

    // Enable 3D depth buffer
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    // Point smoothing for particle visualization
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);

    // Allocate VBO for Vector3f particles
    glGenBuffers(1, &particleVBO);
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLES * sizeof(Vector3f), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Register VBO with CUDA Interop
    cudaGraphicsGLRegisterBuffer(&particleCudaResource, particleVBO, cudaGraphicsMapFlagsWriteDiscard);
}

void GLRenderer::resizeViewport(int w, int h) {
    if (h == 0) h = 1;
    float aspect = static_cast<float>(w) / static_cast<float>(h);

    glViewport(0, 0, w, h);

    // Set 3D Perspective Projection Matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    customPerspective(45.0f, aspect, 0.1f, 500.0f); // Replaced gluPerspective

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
}

void GLRenderer::render(const Simulation& sim) {
    // Clear both color and depth buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Position 3D Camera to view the center of the grid domain
    float centerX = X_GRID * 0.5f;
    float centerY = Y_GRID * 0.5f;
    float centerZ = Z_GRID * 0.5f;
    float maxDim = std::max({ static_cast<float>(X_GRID), static_cast<float>(Y_GRID), static_cast<float>(Z_GRID) });

    // Replaced gluLookAt
    customLookAt(
        centerX + maxDim, centerY + maxDim , centerZ + maxDim, // Camera Position
        centerX, centerY, centerZ,                                                 // Target Center
        0.0f, 1.0f, 0.0f                                                           // Up Vector
    );

    renderBackgroundGrid();
    renderColliders(sim.getCollisionManager());

    const auto& particles = sim.getParticles();

    if (particles.water.num_particles > 0) {
        glColor3f(0.2f, 0.6f, 1.0f); // Blue for water
        renderParticles(particles.water.d_Xp, particles.water.num_particles);
    }
    if (particles.snow.num_particles > 0) {
        glColor3f(0.9f, 0.9f, 0.9f); // White for snow
        renderParticles(particles.snow.d_Xp, particles.snow.num_particles);
    }
    if (particles.elastic.num_particles > 0) {
        glColor3f(0.8f, 0.2f, 0.2f); // Red for elastic
        renderParticles(particles.elastic.d_Xp, particles.elastic.num_particles);
    }
}

void GLRenderer::renderParticles(const Vector3f* d_particles, int count) {
    if (count == 0 || !d_particles) return;

    // Map CUDA resource to VBO pointer
    cudaGraphicsMapResources(1, &particleCudaResource, 0);
    Vector3f* d_vbo_ptr;
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_ptr, &num_bytes, particleCudaResource);

    // Copy 3D particle positions from CUDA to GL VBO
    cudaMemcpy(d_vbo_ptr, d_particles, count * sizeof(Vector3f), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &particleCudaResource, 0);

    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    glEnableClientState(GL_VERTEX_ARRAY);

    // 3 components per vertex (X, Y, Z) for 3D Vector3f
    glVertexPointer(3, GL_FLOAT, sizeof(Vector3f), (void*)0);

    glPointSize(3.5f);
    glDrawArrays(GL_POINTS, 0, count);

    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GLRenderer::renderBackgroundGrid() {
    // 3D wireframe collider box that matches the inner collider (because walls need to have a thickness)
    glColor3f(0.3f, 0.3f, 0.3f);
    glLineWidth(1.0f);

    float pad = 2.0f;
    float x_min = pad;
    float x_max = static_cast<float>(X_GRID) - pad;
    float y_min = pad;
    float y_max = static_cast<float>(Y_GRID) - pad;
    float z_min = pad;
    float z_max = static_cast<float>(Z_GRID) - pad;

    glBegin(GL_LINES);
    glVertex3f(x_min, y_min, z_min); glVertex3f(x_max, y_min, z_min);
    glVertex3f(x_max, y_min, z_min); glVertex3f(x_max, y_min, z_max);
    glVertex3f(x_max, y_min, z_max); glVertex3f(x_min, y_min, z_max);
    glVertex3f(x_min, y_min, z_max); glVertex3f(x_min, y_min, z_min);

    glVertex3f(x_min, y_max, z_min); glVertex3f(x_max, y_max, z_min);
    glVertex3f(x_max, y_max, z_min); glVertex3f(x_max, y_max, z_max);
    glVertex3f(x_max, y_max, z_max); glVertex3f(x_min, y_max, z_max);
    glVertex3f(x_min, y_max, z_max); glVertex3f(x_min, y_max, z_min);

    glVertex3f(x_min, y_min, z_min); glVertex3f(x_min, y_max, z_min);
    glVertex3f(x_max, y_min, z_min); glVertex3f(x_max, y_max, z_min);
    glVertex3f(x_max, y_min, z_max); glVertex3f(x_max, y_max, z_max);
    glVertex3f(x_min, y_min, z_max); glVertex3f(x_min, y_max, z_max);
    glEnd();
}

void GLRenderer::renderColliders(const CollisionManager& collisionManager) {
    glColor3f(0.5f, 0.1f, 0.1f);

    for (const auto& obj : collisionManager.h_objects) {
        if (obj.type == 1) { // 3D Box Wireframe
            float cx = obj.center.x, cy = obj.center.y, cz = obj.center.z;
            float hx = obj.size.x, hy = obj.size.y, hz = obj.size.z;

            // skip rendering the physical domain walls, only render wireframe
            bool isDomainWall = (hy == Y_GRID * 0.5f && hz == Z_GRID * 0.5f) ||
                (hx == X_GRID * 0.5f && hz == Z_GRID * 0.5f) ||
                (hx == X_GRID * 0.5f && hy == Y_GRID * 0.5f);

            if (isDomainWall) continue;

            glBegin(GL_LINES);
            glVertex3f(cx - hx, cy - hy, cz - hz); glVertex3f(cx + hx, cy - hy, cz - hz);
            glVertex3f(cx + hx, cy - hy, cz - hz); glVertex3f(cx + hx, cy + hy, cz - hz);
            glVertex3f(cx + hx, cy + hy, cz - hz); glVertex3f(cx - hx, cy + hy, cz - hz);
            glVertex3f(cx - hx, cy + hy, cz - hz); glVertex3f(cx - hx, cy - hy, cz - hz);

            glVertex3f(cx - hx, cy - hy, cz + hz); glVertex3f(cx + hx, cy - hy, cz + hz);
            glVertex3f(cx + hx, cy - hy, cz + hz); glVertex3f(cx + hx, cy + hy, cz + hz);
            glVertex3f(cx + hx, cy + hy, cz + hz); glVertex3f(cx - hx, cy + hy, cz + hz);
            glVertex3f(cx - hx, cy + hy, cz + hz); glVertex3f(cx - hx, cy - hy, cz + hz);

            glVertex3f(cx - hx, cy - hy, cz - hz); glVertex3f(cx - hx, cy - hy, cz + hz);
            glVertex3f(cx + hx, cy - hy, cz - hz); glVertex3f(cx + hx, cy - hy, cz + hz);
            glVertex3f(cx + hx, cy + hy, cz - hz); glVertex3f(cx + hx, cy + hy, cz + hz);
            glVertex3f(cx - hx, cy + hy, cz - hz); glVertex3f(cx - hx, cy + hy, cz + hz);
            glEnd();
        }
        else if (obj.type == 0) { // 3D Sphere Wireframe Approximation
            float cx = obj.center.x, cy = obj.center.y, cz = obj.center.z;
            float r = obj.size.x;

            const int stacks = 10;
            const int sectors = 16;
            const float PI = 3.1415926535f;

            for (int i = 1; i < stacks; ++i) {
                float phi = PI * static_cast<float>(i) / static_cast<float>(stacks);
                float ring_r = r * sinf(phi);
                float ring_y = r * cosf(phi);

                glBegin(GL_LINE_LOOP);
                for (int j = 0; j < sectors; ++j) {
                    float theta = 2.0f * PI * static_cast<float>(j) / static_cast<float>(sectors);
                    float x = ring_r * cosf(theta);
                    float z = ring_r * sinf(theta);
                    glVertex3f(cx + x, cy + ring_y, cz + z);
                }
                glEnd();
            }

            for (int j = 0; j < sectors; ++j) {
                float theta = 2.0f * PI * static_cast<float>(j) / static_cast<float>(sectors);

                glBegin(GL_LINE_STRIP);
                for (int i = 0; i <= stacks; ++i) {
                    float phi = PI * static_cast<float>(i) / static_cast<float>(stacks);
                    float x = r * sinf(phi) * cosf(theta);
                    float y = r * cosf(phi);
                    float z = r * sinf(phi) * sinf(theta);
                    glVertex3f(cx + x, cy + y, cz + z);
                }
                glEnd();
            }
        }
    }
}

void GLRenderer::freeGL() {
    if (particleCudaResource) {
        cudaGraphicsUnregisterResource(particleCudaResource);
        particleCudaResource = nullptr;
    }
    if (particleVBO) {
        glDeleteBuffers(1, &particleVBO);
        particleVBO = 0;
    }
}