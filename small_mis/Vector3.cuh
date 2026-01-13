#pragma once 
#include <iostream>
#include <cuda_runtime.h>

class Vector3
{
public:
    __host__ __device__ Vector3() {}
    __host__ __device__ Vector3(float v) : e{v, v, v} {}
    __host__ __device__ Vector3(float e0, float e1, float e2)
    {
        e[0] = e0;
        e[1] = e1;
        e[2] = e2;
    }
    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }
    __host__ __device__ inline float r() const { return e[0]; }
    __host__ __device__ inline float g() const { return e[1]; }
    __host__ __device__ inline float b() const { return e[2]; }

    __host__ __device__ inline const Vector3 &operator+() const { return *this; }
    __host__ __device__ inline Vector3 operator-() const { return Vector3(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline float operator[](int i) const { return e[i]; }
    __host__ __device__ inline float &operator[](int i) { return e[i]; };

    __host__ __device__ inline Vector3 &operator+=(const Vector3 &v2);
    __host__ __device__ inline Vector3 &operator-=(const Vector3 &v2);
    __host__ __device__ inline Vector3 &operator*=(const Vector3 &v2);
    __host__ __device__ inline Vector3 &operator/=(const Vector3 &v2);
    __host__ __device__ inline Vector3 &operator*=(const float t);
    __host__ __device__ inline Vector3 &operator/=(const float t);

    __host__ __device__ inline float Length() const { return sqrtf(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); }
    __host__ __device__ inline float SquareLength() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }
    __host__ __device__ inline void Normalize();

    float e[3];
};

inline std::istream &operator>>(std::istream &is, Vector3 &t)
{
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

inline std::ostream &operator<<(std::ostream &os, const Vector3 &t)
{
    os << t.e[0] << " " << t.e[1] << " " << t.e[2];
    return os;
}

__host__ __device__ inline void Vector3::Normalize()
{
    float k = 1.0 / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
}

__host__ __device__ inline Vector3 operator+(const Vector3 &v1, const Vector3 &v2)
{
    return Vector3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline Vector3 operator-(const Vector3 &v1, const Vector3 &v2)
{
    return Vector3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline Vector3 operator*(const Vector3 &v1, const Vector3 &v2)
{
    return Vector3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline Vector3 operator/(const Vector3 &v1, const Vector3 &v2)
{
    return Vector3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline Vector3 operator*(float t, const Vector3 &v)
{
    return Vector3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline Vector3 operator/(Vector3 v, float t)
{
    return Vector3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

__host__ __device__ inline Vector3 operator*(const Vector3 &v, float t)
{
    return Vector3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline float Dot(const Vector3 &v1, const Vector3 &v2)
{
    return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

__host__ __device__ inline Vector3 Cross(const Vector3 &v1, const Vector3 &v2)
{
    return Vector3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
                (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
                (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}

__host__ __device__ inline Vector3 &Vector3::operator+=(const Vector3 &v)
{
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

__host__ __device__ inline Vector3 &Vector3::operator*=(const Vector3 &v)
{
    e[0] *= v.e[0];
    e[1] *= v.e[1];
    e[2] *= v.e[2];
    return *this;
}

__host__ __device__ inline Vector3 &Vector3::operator/=(const Vector3 &v)
{
    e[0] /= v.e[0];
    e[1] /= v.e[1];
    e[2] /= v.e[2];
    return *this;
}

__host__ __device__ inline Vector3 &Vector3::operator-=(const Vector3 &v)
{
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
}

__host__ __device__ inline Vector3 &Vector3::operator*=(const float t)
{
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

__host__ __device__ inline Vector3 &Vector3::operator/=(const float t)
{
    float k = 1.0 / t;

    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
    return *this;
}

__host__ __device__ inline Vector3 Normalize(Vector3 v)
{
    return v / v.Length();
}