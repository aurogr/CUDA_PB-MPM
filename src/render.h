// GLRenderer.h
#pragma once
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "simulation.h"

class GLRenderer {
private:
    GLuint particleVBO = 0;
    cudaGraphicsResource_t particleCudaResource = nullptr;

public:
    GLRenderer();
    ~GLRenderer();

    void initializeGL();
    void resizeViewport(int width, int height);
    void render(const Simulation& sim);
    void freeGL();

private:
    void renderBackgroundGrid();
    void renderColliders(const CollisionManager& collisionManager);
    void renderParticles(const Vector2f* d_particles, int count);
};