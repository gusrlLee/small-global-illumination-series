#ifndef __TINY_PATH_TRACING_STRUCTS_HEADER__
#define __TINY_PATH_TRACING_STRUCTS_HEADER__

#include <vector_types.h>
#include <optix.h>

struct MaterialData 
{
    float3 diffuse;
    float3 emission;
};

struct Params
{
    uchar4* image;

    float4* accum_buffer;

    unsigned int width;
    unsigned int height;
    unsigned int frame_index;

    OptixTraversableHandle handle; // acceleration structure handle

    float3* vertices;
    uint3* indices;

    MaterialData* materials;
    unsigned int* matIndices;
};

struct RayPayload
{
    float3 radiance;    // 현재까지 누적된 빛의 양
    float3 throughput;  // 빛이 감쇠되는 비율 (반사율의 곱)
    bool done;          // 추적 종료 여부
    unsigned int seed;  // 랜덤 시드
    
    // 다음 레이를 위한 준비
    float3 origin;
    float3 direction;
};

#endif // __TINY_PATH_TRACING_STRUCTS_HEADER__