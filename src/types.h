#pragma once

#include <cuda_runtime.h>
#include <math.h>

struct alignas(16) Vector3f {
    float x, y, z;

    // Constructors
    __host__ __device__ Vector3f() : x(0.0f), y(0.0f), z(0.0f) {}
    __host__ __device__ Vector3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    // Interop with CUDA float3
    __host__ __device__ Vector3f(const float3& f) : x(f.x), y(f.y), z(f.z) {}
    __host__ __device__ operator float3() const { return make_float3(x, y, z); }

    // Operators
    __host__ __device__ inline Vector3f operator-() const { return Vector3f(-x, -y, -z); }

    __host__ __device__ inline Vector3f operator+(const Vector3f& v) const { return Vector3f(x + v.x, y + v.y, z + v.z); }
    __host__ __device__ inline Vector3f operator-(const Vector3f& v) const { return Vector3f(x - v.x, y - v.y, z - v.z); }

    __host__ __device__ inline Vector3f operator*(float s) const { return Vector3f(x * s, y * s, z * s); }
    __host__ __device__ inline Vector3f operator/(float s) const { float inv = 1.0f / s; return Vector3f(x * inv, y * inv, z * inv); }

    __host__ __device__ inline Vector3f& operator+=(const Vector3f& v) { x += v.x; y += v.y; z += v.z; return *this; }
    __host__ __device__ inline Vector3f& operator-=(const Vector3f& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }

    __host__ __device__ inline Vector3f& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
    __host__ __device__ inline Vector3f& operator/=(float s) { float inv = 1.0f / s; x *= inv; y *= inv; z *= inv; return *this; }

    __host__ __device__ inline bool operator==(const Vector3f& v) const { return x == v.x && y == v.y && z == v.z; }
    __host__ __device__ inline bool operator!=(const Vector3f& v) const { return !(*this == v); }

    // Methods
    __host__ __device__ inline float dot(const Vector3f& v) const { return x * v.x + y * v.y + z * v.z; }
    __host__ __device__ inline Vector3f cross(const Vector3f& v) const {
        return Vector3f(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }
    __host__ __device__ inline float length_sq() const { return x * x + y * y + z * z; }
    __host__ __device__ inline float length() const { return sqrtf(length_sq()); }
    __host__ __device__ inline Vector3f normalized() const {
        float len = length();
        return len > 1e-8f ? (*this) / len : Vector3f(0.0f, 0.0f, 0.0f);
    }

    __host__ __device__ Vector3f clamp(float min_val, float max_val) const {
        return Vector3f(
            fminf(fmaxf(x, min_val), max_val),
            fminf(fmaxf(y, min_val), max_val),
            fminf(fmaxf(z, min_val), max_val)
        );
    }
};

__host__ __device__ inline Vector3f operator*(float s, const Vector3f& v) {
    return v * s;
}

__host__ __device__ inline Vector3f operator+(const Vector3f& v, float s) {
    return Vector3f(v.x + s, v.y + s, v.z + s);
}

struct alignas(16) Matrix3f {
    float m00, m01, m02;
    float m10, m11, m12;
    float m20, m21, m22;

    // Constructors
    __host__ __device__ Matrix3f()
        : m00(0.0f), m01(0.0f), m02(0.0f),
        m10(0.0f), m11(0.0f), m12(0.0f),
        m20(0.0f), m21(0.0f), m22(0.0f) {
    }

    __host__ __device__ Matrix3f(float m00, float m01, float m02,
        float m10, float m11, float m12,
        float m20, float m21, float m22)
        : m00(m00), m01(m01), m02(m02),
        m10(m10), m11(m11), m12(m12),
        m20(m20), m21(m21), m22(m22) {
    }

    // Factory Helpers
    __host__ __device__ static inline Matrix3f outer_product(const Vector3f& a, const Vector3f& b) {
        return Matrix3f(
            a.x * b.x, a.x * b.y, a.x * b.z,
            a.y * b.x, a.y * b.y, a.y * b.z,
            a.z * b.x, a.z * b.y, a.z * b.z
        );
    }

    // Operators
    __host__ __device__ inline Matrix3f operator-() const {
        return Matrix3f(
            -m00, -m01, -m02,
            -m10, -m11, -m12,
            -m20, -m21, -m22
        );
    }

    __host__ __device__ inline Matrix3f operator+(const Matrix3f& m) const {
        return Matrix3f(
            m00 + m.m00, m01 + m.m01, m02 + m.m02,
            m10 + m.m10, m11 + m.m11, m12 + m.m12,
            m20 + m.m20, m21 + m.m21, m22 + m.m22
        );
    }

    __host__ __device__ inline Matrix3f operator-(const Matrix3f& m) const {
        return Matrix3f(
            m00 - m.m00, m01 - m.m01, m02 - m.m02,
            m10 - m.m10, m11 - m.m11, m12 - m.m12,
            m20 - m.m20, m21 - m.m21, m22 - m.m22
        );
    }

    __host__ __device__ inline Matrix3f operator*(const Matrix3f& m) const {
        return Matrix3f(
            m00 * m.m00 + m01 * m.m10 + m02 * m.m20,
            m00 * m.m01 + m01 * m.m11 + m02 * m.m21,
            m00 * m.m02 + m01 * m.m12 + m02 * m.m22,

            m10 * m.m00 + m11 * m.m10 + m12 * m.m20,
            m10 * m.m01 + m11 * m.m11 + m12 * m.m21,
            m10 * m.m02 + m11 * m.m12 + m12 * m.m22,

            m20 * m.m00 + m21 * m.m10 + m22 * m.m20,
            m20 * m.m01 + m21 * m.m11 + m22 * m.m21,
            m20 * m.m02 + m21 * m.m12 + m22 * m.m22
        );
    }

    __host__ __device__ inline Vector3f operator*(const Vector3f& v) const {
        return Vector3f(
            m00 * v.x + m01 * v.y + m02 * v.z,
            m10 * v.x + m11 * v.y + m12 * v.z,
            m20 * v.x + m21 * v.y + m22 * v.z
        );
    }

    __host__ __device__ inline Matrix3f operator*(float s) const {
        return Matrix3f(
            m00 * s, m01 * s, m02 * s,
            m10 * s, m11 * s, m12 * s,
            m20 * s, m21 * s, m22 * s
        );
    }

    __host__ __device__ inline Matrix3f operator/(float s) const {
        float inv = 1.0f / s;
        return Matrix3f(
            m00 * inv, m01 * inv, m02 * inv,
            m10 * inv, m11 * inv, m12 * inv,
            m20 * inv, m21 * inv, m22 * inv
        );
    }

    __host__ __device__ inline Matrix3f& operator+=(const Matrix3f& m) {
        m00 += m.m00; m01 += m.m01; m02 += m.m02;
        m10 += m.m10; m11 += m.m11; m12 += m.m12;
        m20 += m.m20; m21 += m.m21; m22 += m.m22;
        return *this;
    }

    __host__ __device__ inline Matrix3f& operator-=(const Matrix3f& m) {
        m00 -= m.m00; m01 -= m.m01; m02 -= m.m02;
        m10 -= m.m10; m11 -= m.m11; m12 -= m.m12;
        m20 -= m.m20; m21 -= m.m21; m22 -= m.m22;
        return *this;
    }

    __host__ __device__ inline Matrix3f& operator*=(float s) {
        m00 *= s; m01 *= s; m02 *= s;
        m10 *= s; m11 *= s; m12 *= s;
        m20 *= s; m21 *= s; m22 *= s;
        return *this;
    }

    __host__ __device__ inline Matrix3f& operator*=(const Matrix3f& m) {
        *this = *this * m;
        return *this;
    }

    // Methods
    __host__ __device__ inline float trace() const { return m00 + m11 + m22; }

    __host__ __device__ inline float det() const {
        return m00 * (m11 * m22 - m12 * m21)
            - m01 * (m10 * m22 - m12 * m20)
            + m02 * (m10 * m21 - m11 * m20);
    }

    __host__ __device__ inline Matrix3f transpose() const {
        return Matrix3f(
            m00, m10, m20,
            m01, m11, m21,
            m02, m12, m22
        );
    }

    __host__ __device__ inline Matrix3f inverse() const {
        float d = det();
        float invDet = (fabsf(d) > 1e-8f) ? (1.0f / d) : 0.0f;

        return Matrix3f(
            (m11 * m22 - m12 * m21) * invDet,
            -(m01 * m22 - m02 * m21) * invDet,
            (m01 * m12 - m02 * m11) * invDet,

            -(m10 * m22 - m12 * m20) * invDet,
            (m00 * m22 - m02 * m20) * invDet,
            -(m00 * m12 - m02 * m10) * invDet,

            (m10 * m21 - m11 * m20) * invDet,
            -(m00 * m21 - m01 * m20) * invDet,
            (m00 * m11 - m01 * m10) * invDet
        );
    }

    // Iterative 3D Polar Decomposition (Newton-Raphson)
    __host__ __device__ void polar_decomp(Matrix3f* R, Matrix3f* S) const {
        Matrix3f CurrR = *this;
        for (int i = 0; i < 6; ++i) {
            Matrix3f invRTrans = CurrR.inverse().transpose();
            CurrR = (CurrR + invRTrans) * 0.5f;
        }
        *R = CurrR;
        *S = R->transpose() * (*this);
    }

    // 3D SVD using Polar Decomposition + Jacobi Rotations
    __host__ __device__ void svd(Matrix3f* U, Vector3f* Sigma, Matrix3f* V) const {
        Matrix3f R, S;
        polar_decomp(&R, &S);

        Matrix3f V_mat = Matrix3f(1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f);
        Matrix3f S_mat = S;

        for (int iter = 0; iter < 5; ++iter) {
            auto jacobi_pair = [&](int p, int q) {
                float spq;
                if (p == 0 && q == 1) spq = S_mat.m01;
                else if (p == 0 && q == 2) spq = S_mat.m02;
                else spq = S_mat.m12;

                if (fabsf(spq) < 1e-6f) return;

                float spp = (p == 0) ? S_mat.m00 : ((p == 1) ? S_mat.m11 : S_mat.m22);
                float sqq = (q == 0) ? S_mat.m00 : ((q == 1) ? S_mat.m11 : S_mat.m22);

                float tau = (sqq - spp) / (2.0f * spq);
                float t = (tau >= 0.0f) ? (1.0f / (tau + sqrtf(1.0f + tau * tau)))
                    : (1.0f / (tau - sqrtf(1.0f + tau * tau)));
                float c = 1.0f / sqrtf(1.0f + t * t);
                float s = t * c;

                Matrix3f J(1.0f, 0.0f, 0.0f,
                    0.0f, 1.0f, 0.0f,
                    0.0f, 0.0f, 1.0f);

                if (p == 0 && q == 1) { J.m00 = c; J.m01 = s; J.m10 = -s; J.m11 = c; }
                else if (p == 0 && q == 2) { J.m00 = c; J.m02 = s; J.m20 = -s; J.m22 = c; }
                else { J.m11 = c; J.m12 = s; J.m21 = -s; J.m22 = c; }

                S_mat = J.transpose() * S_mat * J;
                V_mat = V_mat * J;
                };

            jacobi_pair(0, 1);
            jacobi_pair(0, 2);
            jacobi_pair(1, 2);
        }

        *Sigma = Vector3f(S_mat.m00, S_mat.m11, S_mat.m22);
        *V = V_mat;
        *U = R * V_mat;
    }

    __host__ __device__ Matrix3f diag_product(const Vector3f& diag) const {
        return Matrix3f(
            m00 * diag.x, m01 * diag.y, m02 * diag.z,
            m10 * diag.x, m11 * diag.y, m12 * diag.z,
            m20 * diag.x, m21 * diag.y, m22 * diag.z
        );
    }

    __host__ __device__ Matrix3f diag_product_inv(const Vector3f& diag) const {
        return Matrix3f(
            m00 / diag.x, m01 / diag.y, m02 / diag.z,
            m10 / diag.x, m11 / diag.y, m12 / diag.z,
            m20 / diag.x, m21 / diag.y, m22 / diag.z
        );
    }
};

__host__ __device__ static inline Matrix3f identity() {
    return Matrix3f(
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    );
}

__host__ __device__ static inline Matrix3f outer_product(const Vector3f& a, const Vector3f& b) {
    return Matrix3f(
        a.x * b.x, a.x * b.y, a.x * b.z,
        a.y * b.x, a.y * b.y, a.y * b.z,
        a.z * b.x, a.z * b.y, a.z * b.z
    );
}

__host__ __device__ static inline Matrix3f operator*(const float& a, const Matrix3f& b) {
    return Matrix3f(
        a * b.m00, a * b.m01, a * b.m02,
        a * b.m10, a * b.m11, a * b.m12,
        a * b.m20, a * b.m21, a * b.m22
    );
}