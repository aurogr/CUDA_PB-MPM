#pragma once

#include <cuda_runtime.h>
#include <math.h>

struct alignas(8) Vector2f {
    float x, y;

    // Constructors
    __host__ __device__ Vector2f() : x(0.0f), y(0.0f) {}
    __host__ __device__ Vector2f(float x_, float y_) : x(x_), y(y_) {}

    // Interop with CUDA float2
    __host__ __device__ Vector2f(const float2& f) : x(f.x), y(f.y) {}
    __host__ __device__ operator float2() const { return make_float2(x, y); }

    // Operators
    __host__ __device__ inline Vector2f operator-() const { return Vector2f(-x, -y); }

    __host__ __device__ inline Vector2f operator+(const Vector2f& v) const { return Vector2f(x + v.x, y + v.y); }
    __host__ __device__ inline Vector2f operator-(const Vector2f& v) const { return Vector2f(x - v.x, y - v.y); }

    __host__ __device__ inline Vector2f operator*(float s) const { return Vector2f(x * s, y * s); }
    __host__ __device__ inline Vector2f operator/(float s) const { float inv = 1.0f / s; return Vector2f(x * inv, y * inv); }

    __host__ __device__ inline Vector2f& operator+=(const Vector2f& v) { x += v.x; y += v.y; return *this; }
    __host__ __device__ inline Vector2f& operator-=(const Vector2f& v) { x -= v.x; y -= v.y; return *this; }

    __host__ __device__ inline Vector2f& operator*=(float s) { x *= s; y *= s; return *this; }
    __host__ __device__ inline Vector2f& operator/=(float s) { float inv = 1.0f / s; x *= inv; y *= inv; return *this; }

    __host__ __device__ inline bool operator==(const Vector2f& v) const { return x == v.x && y == v.y; }
    __host__ __device__ inline bool operator!=(const Vector2f& v) const { return !(*this == v); }

    // Methods
    __host__ __device__ inline float dot(const Vector2f& v) const { return x * v.x + y * v.y; }
    __host__ __device__ inline float cross(const Vector2f& v) const { return x * v.y - y * v.x; } // 2D cross -> scalar z
    __host__ __device__ inline float length_sq() const { return x * x + y * y; }
    __host__ __device__ inline float length() const { return sqrtf(length_sq()); }
    __host__ __device__ inline Vector2f normalized() const {
        float len = length();
        return len > 1e-8f ? (*this) / len : Vector2f(0.0f, 0.0f);
    }
};

__host__ __device__ inline Vector2f operator*(float s, const Vector2f& v) {
    return v * s; // Reutiliza el operador Vector2f * float que ya tienes
}

struct alignas(16) Matrix2f {
    float x, y, z, w; // Memory layout: x=M00, y=M01, z=M10, w=M11

    // Constructors
    __host__ __device__ Matrix2f() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
    __host__ __device__ Matrix2f(float m00, float m01, float m10, float m11)
        : x(m00), y(m01), z(m10), w(m11) {}

    // Interop with CUDA float4
    __host__ __device__ Matrix2f(const float4& f) : x(f.x), y(f.y), z(f.z), w(f.w) {}
    __host__ __device__ operator float4() const { return make_float4(x, y, z, w); }

    // Factory Helpers
    __host__ __device__ static inline Matrix2f identity() { return Matrix2f(1.0f, 0.0f, 0.0f, 1.0f); }
    __host__ __device__ static inline Matrix2f outer_product(const Vector2f& a, const Vector2f& b) {
        return Matrix2f(a.x * b.x, a.x * b.y, a.y * b.x, a.y * b.y);
    }

    // Operators
    __host__ __device__ inline Matrix2f operator-() const { return Matrix2f(-x, -y, -z, -w); }

    __host__ __device__ inline Matrix2f operator+(const Matrix2f& m) const { return Matrix2f(x + m.x, y + m.y, z + m.z, w + m.w); }
    __host__ __device__ inline Matrix2f operator-(const Matrix2f& m) const { return Matrix2f(x - m.x, y - m.y, z - m.z, w - m.w); }

    __host__ __device__ inline Matrix2f operator*(const Matrix2f& m) const {
        return Matrix2f(
            x * m.x + y * m.z, x * m.y + y * m.w,
            z * m.x + w * m.z, z * m.y + w * m.w
        );
    }

    __host__ __device__ inline Vector2f operator*(const Vector2f& v) const {
        return Vector2f(x * v.x + y * v.y, z * v.x + w * v.y);
    }

    __host__ __device__ inline Matrix2f operator*(float s) const { return Matrix2f(x * s, y * s, z * s, w * s); }
    __host__ __device__ inline Matrix2f operator/(float s) const { float inv = 1.0f / s; return Matrix2f(x * inv, y * inv, z * inv, w * inv); }

    __host__ __device__ inline Matrix2f& operator+=(const Matrix2f& m) { x += m.x; y += m.y; z += m.z; w += m.w; return *this; }
    __host__ __device__ inline Matrix2f& operator-=(const Matrix2f& m) { x -= m.x; y -= m.y; z -= m.z; w -= m.w; return *this; }
    __host__ __device__ inline Matrix2f& operator*=(float s) { x *= s; y *= s; z *= s; w *= s; return *this; }
    __host__ __device__ inline Matrix2f& operator*=(const Matrix2f& m) { *this = *this * m; return *this; }

    // Methods
    __host__ __device__ inline float trace() const { return x + w; }
    __host__ __device__ inline float det() const { return x * w - y * z; }
    __host__ __device__ inline Matrix2f transpose() const { return Matrix2f(x, z, y, w); }

    __host__ __device__ inline Matrix2f inverse() const {
        float d = det();
        float invDet = (fabsf(d) > 1e-8f) ? (1.0f / d) : 0.0f;
        return Matrix2f(w * invDet, -y * invDet, -z * invDet, x * invDet);
    }
};


__host__ __device__ static inline Matrix2f outer_product(const Vector2f& a, const Vector2f& b) {
    return Matrix2f(
        a.x * b.x, a.x * b.y,
        a.y * b.x, a.y * b.y
    );
}

__host__ __device__ static inline Matrix2f operator*(const float& a, const Matrix2f& b) {
    return Matrix2f(
        a * b.x, a * b.y, a * b.z, a * b.w
    );
}