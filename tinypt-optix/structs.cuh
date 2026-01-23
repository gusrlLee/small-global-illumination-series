#ifndef __TINY_PATH_TRACING_STRUCTS_HEADER__
#define __TINY_PATH_TRACING_STRUCTS_HEADER__

#include <vector_types.h>
#include <optix.h>

struct Params
{
    uchar4* image;
    unsigned int width;
    unsigned int height;
    OptixTraversableHandle handle; // acceleration structure handle
};

struct RayPayload
{
    float3 result;
};

struct MissData
{
    float3 bg_color;
};

struct HitGroupData
{

};

#endif // __TINY_PATH_TRACING_STRUCTS_HEADER__