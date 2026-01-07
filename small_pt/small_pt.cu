#include <iostream>
#include <cuda_runtime.h>

#include <vector_types.h>
#include <device_launch_parameters.h>

using Vertex = float3;

class Vector3 
{
    public:
        float x, y, z;
        __device__ Vector3() : x(0), y(0), z(0) {}
        __device__ Vector3(float x, float y, float z) : x(x), y(y), z(z) {}
        __device__ Vector3 operator+(const Vector3& other) const { return Vector3(x + other.x, y + other.y, z + other.z); }
        __device__ Vector3 operator-(const Vector3& other) const { return Vector3(x - other.x, y - other.y, z - other.z); }
};

class Ray 
{
    public:
        __device__ Ray() {}
    private:

};


__global__ void hello_kernel()
{
    printf("Hello from the GPU!\\n");
}

int main()
{
    hello_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}