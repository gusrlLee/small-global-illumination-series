#include <optix.h>
#include "structs.cuh"

// Parameter
extern "C" {
    __constant__ Params params;
}

// Ray Generation Program
extern "C" __global__ void __raygen__rg() 
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const unsigned int index = idx.y * dim.x + idx.x;

    params.image[index] = make_uchar4(255, 0, 0, 255);
}

// Miss Program
extern "C" __global__ void __miss__ms()
{

}

// Closest Hit Program 
extern "C" __global__ void __closesthit__ch()
{

}

