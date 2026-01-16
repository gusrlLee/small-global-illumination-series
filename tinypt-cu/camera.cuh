#ifndef __TINY_PATH_TRACING_CAMERA_HEADER__
#define __TINY_PATH_TRACING_CAMERA_HEADER__

#include "pch.cuh"
#include "vec3.cuh"
#include "ray.cuh"

struct Camera 
{
    Vec3 origin;
    Vec3 lowerLeftCorner;
    Vec3 horizontal;
    Vec3 vertical;
    
    __host__ __device__ Camera() {}

    __host__ __device__ Camera(Vec3 lookFrom, Vec3 lookAt, Vec3 vup, float vfov, float aspectRatio) 
    {
        float theta = vfov * (PI / 180.0f);
        float h = tan(theta / 2.0f);

        float vHight = 2.0f * h;
        float vWidth = aspectRatio * vHight;

        Vec3 w = normalize(lookFrom - lookAt);
        Vec3 u = normalize(cross(vup, w));
        Vec3 v = cross(w, u);    

        origin = lookFrom;
        horizontal = vWidth * u;
        vertical = vHight * v;

        lowerLeftCorner = origin - horizontal / 2.0f - vertical / 2.0f - w;
    }

    __device__ __host__ Ray getRay(float s, float t) const 
    {
        return Ray(origin, lowerLeftCorner + s * horizontal + t * vertical - origin);
    }
};

#endif // __TINY_PATH_TRACING_CAMERA_HEADER__