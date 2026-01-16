#ifndef __TINY_PATH_TRACING_RAY_HEADER__
#define __TINY_PATH_TRACING_RAY_HEADER__

#include "vec3.cuh"

struct Ray 
{
    Vec3 orig;
    Vec3 dir;

    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const Vec3& origin, const Vec3& direction) : orig(origin), dir(direction) {}

    __host__ __device__ Vec3 at(float t) const { return orig + t * dir; }
};

#endif // __TINY_PATH_TRACING_RAY_HEADER__