#include <optix.h>
#include "structs.cuh"

// Parameter
extern "C" {
    __constant__ Params params;
}

// Ray Generation Program
extern "C" __global__ void __raygen__rg() 
{
// 현재 스레드가 처리 중인 픽셀의 위치 (x, y)
    const uint3 idx = optixGetLaunchIndex();
    
    // 전체 이미지 크기 (width, height)
    const uint3 dim = optixGetLaunchDimensions();

    // 2D 좌표를 1D 배열 인덱스로 변환
    const unsigned int index = idx.y * dim.x + idx.x;

    // 0.0 ~ 1.0 사이의 값으로 정규화 (UV 좌표)
    float u = (float)idx.x / (float)(dim.x - 1);
    float v = (float)idx.y / (float)(dim.y - 1);

    // R 채널은 가로(u), G 채널은 세로(v)에 따라 변하게 설정
    params.image[index] = make_uchar4(
        (unsigned char)(u * 255.0f), 
        (unsigned char)(v * 255.0f), 
        0, 
        255
    );
}

// Miss Program
extern "C" __global__ void __miss__ms()
{

}

// Closest Hit Program 
extern "C" __global__ void __closesthit__ch()
{

}

