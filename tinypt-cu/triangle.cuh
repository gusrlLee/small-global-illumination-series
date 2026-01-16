#ifndef __TINY_PATH_TRACING_TRIANGLE_HEADER__
#define __TINY_PATH_TRACING_TRIANGLE_HEADER__

#include "vec3.cuh"
#include "material.cuh"

struct Triangle 
{
    Vec3 v0, v1, v2; //vertices of triangle
    Vec3 e1, e2; // edges of triangle 
    Vec3 n; // normalized normal vector 
    uint32_t matId; // material index of triangle

    __host__ __device__ Triangle() {}
    __host__ __device__ Triangle(const Vec3& v0, const Vec3& v1, const Vec3& v2) : v0(v0), v1(v1), v2(v2) 
    {
        e1 = v1 - v0; // edge 1
        e2 = v2 - v0; // edge 2
        n = normalize(cross(e1, e2)); // normal vector 
    }
    __host__ __device__ Triangle(const Vec3& vertex0, const Vec3& vertex1, const Vec3& vertex2, const uint32_t& materialId) : v0(vertex0), v1(vertex1), v2(vertex2), matId(materialId) 
    {
        e1 = v1 - v0; // edge 1
        e2 = v2 - v0; // edge 2
        n = normalize(cross(e1, e2)); // normal vector 
    }

    __host__ __device__ Vec3 normal() const { return n; }
    __host__ __device__ Vec3 edge1() const { return e1; }
    __host__ __device__ Vec3 edge2() const { return e2; }
    
    __host__ __device__ Vec3 centroid() const { return (v0 + v1 + v2) / 3.0f; }
    __host__ __device__ float area() const { return length(cross(e1, e2)) / 2.0f; }
};

#endif // __TINY_PATH_TRACING_TRIANGLE_HEADER__