#pragma once

#include <GLFW\glfw3.h>
#include <cuda_runtime.h>
#include <vector>

#include "constants.h"

enum BoundaryType {
	STICKY = 1,
	SEPARATING = 2,
	SLIDING = 3
};

// Simple POD structure for a single line segment boundary.
// Works seamlessly inside CUDA __device__ and __global__ functions.
struct LineSegmentBoundary {
	float2 start;
	float2 end;
	float2 normal;
	int type;
};

struct BoundaryData {
	LineSegmentBoundary* d_borders; // Pointer to device memory
	int count;                      // Number of border segments
};

class BoundaryManager
{
public:

	std::vector<LineSegmentBoundary> h_borders; // Host side border list
	LineSegmentBoundary* d_borders;            // Device side border array
	int count;

    void InitializeDefaultBorders() {
        h_borders.clear();

        float cub = CUB;
        float gridX = static_cast<float>(X_GRID);
        float gridY = static_cast<float>(Y_GRID);

        // Left border
        h_borders.push_back({ make_float2(cub, cub), make_float2(cub, gridY - cub), make_float2(1.0f, 0.0f), SEPARATING });

        // Right border
        h_borders.push_back({ make_float2(gridX - cub, cub), make_float2(gridX - cub, gridY - cub), make_float2(-1.0f, 0.0f), SEPARATING });

        // Bottom border
        h_borders.push_back({ make_float2(cub, cub), make_float2(gridX - cub, cub), make_float2(0.0f, 1.0f), SEPARATING });

        // Top border
        h_borders.push_back({ make_float2(cub, gridY - cub), make_float2(gridX - cub, gridY - cub), make_float2(0.0f, -1.0f), SEPARATING });

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
