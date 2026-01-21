#ifndef __TINY_PATH_TRACING_VECTOR3_HEADER__
#define __TINY_PATH_TRACING_VECTOR3_HEADER__

#include <cmath>
#include <iostream>
#include "cuda_runtime.h"

struct Vec3 
{
    float x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float v) : x(v), y(v), z(v) {}
    __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    __host__ __device__ Vec3 operator-() const { return Vec3(-x, -y, -z); }
    __host__ __device__ Vec3& operator+=(const Vec3 &v) {
        x += v.x; y += v.y; z += v.z;
        return *this;
    }

    __host__ __device__ Vec3& operator*=(const float t) {
        x *= t; y *= t; z *= t;
        return *this;
    }

    __host__ __device__ Vec3& operator/=(const float t) {
        return *this *= (1.0f / t);
    }

    __host__ __device__ float lengthSq() const {
        return x * x + y * y + z * z;
    }

    __host__ __device__ float length() const {
        return sqrtf(lengthSq()); 
    }
};

__host__ __device__ inline Vec3 operator+(const Vec3 &u, const Vec3 &v) {
    return Vec3(u.x + v.x, u.y + v.y, u.z + v.z);
}

__host__ __device__ inline Vec3 operator-(const Vec3 &u, const Vec3 &v) {
    return Vec3(u.x - v.x, u.y - v.y, u.z - v.z);
}

__host__ __device__ inline Vec3 operator*(const Vec3 &u, const Vec3 &v) {
    return Vec3(u.x * v.x, u.y * v.y, u.z * v.z);
}

__host__ __device__ inline Vec3 operator*(float t, const Vec3 &v) {
    return Vec3(t * v.x, t * v.y, t * v.z);
}

__host__ __device__ inline Vec3 operator*(const Vec3 &v, float t) {
    return t * v;
}

__host__ __device__ inline Vec3 operator/(Vec3 v, float t) {
    return (1.0f / t) * v;
}

__host__ __device__ inline float dot(const Vec3 &u, const Vec3 &v) {
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

__host__ __device__ inline Vec3 cross(const Vec3 &u, const Vec3 &v) {
    return Vec3(u.y * v.z - u.z * v.y,
                   u.z * v.x - u.x * v.z,
                   u.x * v.y - u.y * v.x);
}

__host__ __device__ inline Vec3 normalize(Vec3 v) {
    return v / v.length();
}

__host__ __device__ inline float length(Vec3 v) {
    return v.length();
}

__host__ __device__ inline float lengthSq(Vec3 v) {
    return v.lengthSq();
}

#endif // __TINY_PATH_TRACING_VECTOR3_HEADER__