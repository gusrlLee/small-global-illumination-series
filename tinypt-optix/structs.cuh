#ifndef __TINY_PATH_TRACING_STRUCTS_HEADER__
#define __TINY_PATH_TRACING_STRUCTS_HEADER__

#include <vector_types.h>

struct Params
{
    uchar4* image;
    unsigned int width;
    unsigned int height;
};

struct RayPayload
{
    float3 result;
};

#endif // __TINY_PATH_TRACING_STRUCTS_HEADER__