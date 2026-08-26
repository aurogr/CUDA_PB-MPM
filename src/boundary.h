#pragma once

struct CollisionObjectData {
    int type; // 0 = sphere, 1 = box
    Vector2f center;
    Vector2f size;
    float rotation;
    float friction;
};

struct CollisionManagerData {
    CollisionObjectData* d_objects;
    int count;
};

class LevelSetCollisionManager {
public:
    std::vector<CollisionObjectData> h_objects;
    CollisionObjectData* d_objects = nullptr;
    int count = 0;

    void addSphere(Vector2f center, float radius, float friction) {
        h_objects.push_back({ 0, center, Vector2f(radius, radius), 0.0f, friction });
        count = static_cast<int>(h_objects.size());
    }

    void addBox(Vector2f center, Vector2f size, float rotation, float friction) {
        h_objects.push_back({ 1, center, size, rotation, friction });
        count = static_cast<int>(h_objects.size());
    }

    void copyToDevice() {
        if (count == 0) return;
        if (d_objects != nullptr) cudaFree(d_objects);

        size_t bytes = count * sizeof(CollisionObjectData);
        cudaMalloc((void**)&d_objects, bytes);
        cudaMemcpy(d_objects, h_objects.data(), bytes, cudaMemcpyHostToDevice);
    }

    CollisionManagerData getDeviceData() const {
        return CollisionManagerData{ d_objects, count };
    }

    void free() {
        if (d_objects != nullptr) {
            cudaFree(d_objects);
            d_objects = nullptr;
        }
    }
};