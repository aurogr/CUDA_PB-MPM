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
    float m00, m01, m10, m11;

    // Constructors
    __host__ __device__ Matrix2f() : m00(0.0f), m01(0.0f), m10(0.0f), m11(0.0f) {}
    __host__ __device__ Matrix2f(float m00, float m01, float m10, float m11)
        : m00(m00), m01(m01), m10(m10), m11(m11) {
    }

    // Interop with CUDA float4
    __host__ __device__ Matrix2f(const float4& f) : m00(f.x), m01(f.y), m10(f.z), m11(f.w) {}
    __host__ __device__ operator float4() const { return make_float4(m00, m01, m10, m11); }

    // Factory Helpers
    __host__ __device__ static inline Matrix2f outer_product(const Vector2f& a, const Vector2f& b) {
        return Matrix2f(a.x * b.x, a.x * b.y, a.y * b.x, a.y * b.y);
    }

    // Operators
    __host__ __device__ inline Matrix2f operator-() const { return Matrix2f(-m00, -m01, -m10, -m11); }

    __host__ __device__ inline Matrix2f operator+(const Matrix2f& m) const { return Matrix2f(m00 + m.m00, m01 + m.m01, m10 + m.m10, m11 + m.m11); }
    __host__ __device__ inline Matrix2f operator-(const Matrix2f& m) const { return Matrix2f(m00 - m.m00, m01 - m.m01, m10 - m.m10, m11 - m.m11); }

    __host__ __device__ inline Matrix2f operator*(const Matrix2f& m) const {
        return Matrix2f(
            m00 * m.m00 + m01 * m.m10, m00 * m.m01 + m01 * m.m11,
            m10 * m.m00 + m11 * m.m10, m10 * m.m01 + m11 * m.m11
        );
    }

    __host__ __device__ inline Vector2f operator*(const Vector2f& v) const {
        return Vector2f(m00 * v.x + m01 * v.y, m10 * v.x + m11 * v.y);
    }

    __host__ __device__ inline Matrix2f operator*(float s) const { return Matrix2f(m00 * s, m01 * s, m10 * s, m11 * s); }
    __host__ __device__ inline Matrix2f operator/(float s) const { float inv = 1.0f / s; return Matrix2f(m00 * inv, m01 * inv, m10 * inv, m11 * inv); }

    __host__ __device__ inline Matrix2f& operator+=(const Matrix2f& m) { m00 += m.m00; m01 += m.m01; m10 += m.m10; m11 += m.m11; return *this; }
    __host__ __device__ inline Matrix2f& operator-=(const Matrix2f& m) { m00 -= m.m00; m01 -= m.m01; m10 -= m.m10; m11 -= m.m11; return *this; }
    __host__ __device__ inline Matrix2f& operator*=(float s) { m00 *= s; m01 *= s; m10 *= s; m11 *= s; return *this; }
    __host__ __device__ inline Matrix2f& operator*=(const Matrix2f& m) { *this = *this * m; return *this; }

    // Methods
    __host__ __device__ inline float trace() const { return m00 + m11; }
    __host__ __device__ inline float det() const { return m00 * m11 - m01 * m10; }
    __host__ __device__ inline Matrix2f transpose() const { return Matrix2f(m00, m10, m01, m11); }

    __host__ __device__ inline Matrix2f inverse() const {
        float d = det();
        float invDet = (fabsf(d) > 1e-8f) ? (1.0f / d) : 0.0f;
        return Matrix2f(m11 * invDet, -m01 * invDet, -m10 * invDet, m00 * invDet);
    }

    __host__ __device__ void polar_decomp(Matrix2f* R, Matrix2f* S) const {
        float x = m00 + m11; // a + d
        float y = m10 - m01; // c - b
        float scale = sqrtf(x * x + y * y);

        if (scale < 1e-6f) {
            *R = Matrix2f(1.0f, 0.0f, 0.0f, 1.0f);
        }
        else {
            float c = x / scale;
            float s = y / scale;
            *R = Matrix2f(c, -s,
                s, c);
        }

        // S = R^T * F
        *S = R->transpose() * (*this);
    }

    __device__ void svd(Matrix2f* U, Matrix2f* Sigma, Matrix2f* V) const {
        // 1. Compute V and eigenvalues of F^T * F
        Matrix2f FTF = transpose() * (*this);

        float trace = FTF.m00 + FTF.m11;
        float det = FTF.m00 * FTF.m11 - FTF.m01 * FTF.m10;

        float gap = FTF.m00 - FTF.m11;
        float term = sqrtf(fmaxf(0.0f, gap * gap + 4.0f * FTF.m01 * FTF.m01));

        // Singular values (square roots of eigenvalues of F^T * F)
        float s0 = sqrtf(fmaxf(0.0f, 0.5f * (trace + term)));
        float s1 = sqrtf(fmaxf(0.0f, 0.5f * (trace - term)));
        *Sigma = Matrix2f(s0, 0.0f, 0.0f, s1);

        // 2. Compute angle for V
        float angle_v = 0.5f * atan2f(2.0f * FTF.m01, gap);
        float cv = cosf(angle_v);
        float sv = sinf(angle_v);
        *V = Matrix2f(cv, -sv, sv, cv);

        // 3. Compute U = F * V * Sigma^-1
        Matrix2f invSigma(s0 > 1e-6f ? 1.0f / s0 : 0.0f, 0.0f,
            0.0f, s1 > 1e-6f ? 1.0f / s1 : 0.0f);
        *U = (*this) * (*V) * invSigma;
    }
};

__host__ __device__ static inline Matrix2f identity() { return Matrix2f(1.0f, 0.0f, 0.0f, 1.0f); }

__host__ __device__ static inline Matrix2f outer_product(const Vector2f& a, const Vector2f& b) {
    return Matrix2f(
        a.x * b.x, a.x * b.y,
        a.y * b.x, a.y * b.y
    );
}

__host__ __device__ static inline Matrix2f operator*(const float& a, const Matrix2f& b) {
    return Matrix2f(
        a * b.m00, a * b.m01, a * b.m10, a * b.m11
    );
}