#pragma once

#include <GLFW\glfw3.h>
#include <cuda_runtime.h>
#include <vector>

#include "constants.h"

struct LineSegmentBoundary {
	Vector2f start;
    Vector2f end;
    Vector2f normal;
    float friction;
};

struct BoundaryData {
	LineSegmentBoundary* d_borders;
	int count;                      
};

class BoundaryManager
{
public:

    std::vector<LineSegmentBoundary> h_borders;
	LineSegmentBoundary* d_borders;            
	int count;

    void InitializeDefaultBorders() {
        h_borders.clear();

        float cub = 1.5f;
        float gridX = static_cast<float>(X_GRID);
        float gridY = static_cast<float>(Y_GRID);

        // Left border
        h_borders.push_back({ Vector2f(cub, cub), Vector2f(cub, gridY - cub), Vector2f(1.0f, 0.0f), 5.0f });

        // Right border
        h_borders.push_back({ Vector2f(gridX - cub, cub), Vector2f(gridX - cub, gridY - cub), Vector2f(-1.0f, 0.0f), 5.0f });

        // Bottom border
        h_borders.push_back({ Vector2f(cub, cub), Vector2f(gridX - cub, cub), Vector2f(0.0f, 1.0f), 5.0f });

        // Top border
        h_borders.push_back({ Vector2f(cub, gridY - cub), Vector2f(gridX - cub, gridY - cub), Vector2f(0.0f, -1.0f), 5.0f });

        count = static_cast<int>(h_borders.size());
    }

    // Allocates and copies host borders onto the GPU
    void CopyToDevice() {
        if (count == 0) return;

        if (d_borders != nullptr) {
            cudaFree(d_borders);
        }

        size_t bytes = count * sizeof(LineSegmentBoundary);
        cudaMalloc((void**)&d_borders, bytes);
        cudaMemcpy(d_borders, h_borders.data(), bytes, cudaMemcpyHostToDevice);
    }

    // Call this to retrieve a lightweight handle for kernel launches
    BoundaryData GetDeviceData() const {
        return BoundaryData{ d_borders, count };
    }

    void FreeDeviceMemory() {
        if (d_borders != nullptr) {
            cudaFree(d_borders);
            d_borders = nullptr;
        }
    }
};
