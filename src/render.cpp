// GLRenderer.cu
#include "render.h"
#include "constants.h"
#include <cmath>

GLRenderer::GLRenderer() {}

GLRenderer::~GLRenderer() {
    freeGL();
}

void GLRenderer::initializeGL() {
    glewInit();

    glGenBuffers(1, &particleVBO);
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLES * sizeof(Vector2f), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&particleCudaResource, particleVBO, cudaGraphicsMapFlagsWriteDiscard);
}

void GLRenderer::resizeViewport(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, X_GRID, 0, Y_GRID, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
}
void GLRenderer::render(const Simulation& sim) {
    glClear(GL_COLOR_BUFFER_BIT);

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

void GLRenderer::renderParticles(const Vector2f* d_particles, int count) {
    if (count == 0 || !d_particles) return;

    cudaGraphicsMapResources(1, &particleCudaResource, 0);
    Vector2f* d_vbo_ptr;
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_ptr, &num_bytes, particleCudaResource);

    cudaMemcpy(d_vbo_ptr, d_particles, count * sizeof(Vector2f), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &particleCudaResource, 0);

    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_FLOAT, 0, (void*)0);

    glColor3f(0.2f, 0.6f, 1.0f);
    glEnable(GL_POINT_SMOOTH);
    glPointSize(3.0f);
    glDrawArrays(GL_POINTS, 0, count);

    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GLRenderer::renderBackgroundGrid() {
    glColor3f(0.2f, 0.2f, 0.2f);
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    for (int x = 0; x <= X_GRID; ++x) {
        glVertex2f(static_cast<float>(x) * H, 0.0f);
        glVertex2f(static_cast<float>(x) * H, static_cast<float>(Y_GRID) * H);
    }
    for (int y = 0; y <= Y_GRID; ++y) {
        glVertex2f(0.0f, static_cast<float>(y) * H);
        glVertex2f(static_cast<float>(X_GRID) * H, static_cast<float>(y) * H);
    }
    glEnd();
}

void GLRenderer::renderColliders(const CollisionManager& collisionManager) {
    glColor3f(0.5f, 0.0f, 0.0f);
    for (const auto& obj : collisionManager.h_objects) {
        if (obj.type == 1) { // Box
            float cx = obj.center.x, cy = obj.center.y;
            float hx = obj.size.x, hy = obj.size.y;
            glBegin(GL_QUADS);
            glVertex2f(cx - hx, cy - hy);
            glVertex2f(cx + hx, cy - hy);
            glVertex2f(cx + hx, cy + hy);
            glVertex2f(cx - hx, cy + hy);
            glEnd();
        }
        else if (obj.type == 0) { // Sphere
            float cx = obj.center.x, cy = obj.center.y, r = obj.size.x;
            glBegin(GL_TRIANGLE_FAN);
            glVertex2f(cx, cy);
            for (int i = 0; i <= 20; ++i) {
                float theta = 2.0f * 3.1415926f * static_cast<float>(i) / 20.0f;
                glVertex2f(cx + r * cosf(theta), cy + r * sinf(theta));
            }
            glEnd();
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