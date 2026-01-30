#include <optix.h>
#include "structs.cuh"
#include "vec3.cuh"

// Parameter
extern "C"
{
    __constant__ Params params;
}

__forceinline__ __device__ unsigned int lcg(unsigned int &seed)
{
    const unsigned int LCG_A = 1664525u;
    const unsigned int LCG_C = 1013904223u;
    seed = (LCG_A * seed + LCG_C);
    return seed;
}

__forceinline__ __device__ float rnd(unsigned int &seed)
{
    return (float)lcg(seed) / (float)0xFFFFFFFFu;
}

static __forceinline__ __device__ void *unpackPointer(unsigned int i0, unsigned int i1)
{
    const unsigned long long ptr = (unsigned long long)i0 << 32 | i1;
    return (void *)ptr;
}

static __forceinline__ __device__ void packPointer(void *ptr, unsigned int &i0, unsigned int &i1)
{
    const unsigned long long uptr = (unsigned long long)ptr;
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

__forceinline__ __device__ void get_onb(const float3 &n, float3 &b1, float3 &b2)
{
    if (fabs(n.x) > fabs(n.z))
    {
        b1.x = -n.y;
        b1.y = n.x;
        b1.z = 0;
    }
    else
    {
        b1.x = 0;
        b1.y = -n.z;
        b1.z = n.y;
    }
    b1 = normalize(b1);
    b2 = cross(n, b1);
}

__forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3 &p)
{
    const float r = sqrtf(u1);
    const float phi = 2.0f * 3.14159265f * u2;
    p.x = r * cosf(phi);
    p.y = r * sinf(phi);
    p.z = sqrtf(max(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}

// Ray Generation Program
extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const unsigned int pixel_index = idx.y * dim.x + idx.x;

    // 1. 카메라 설정 (Cornell Box View)
    const float3 lookfrom = make_float3(0.0f, 1.0f, 3.4f); // 박스 정면 바깥 (Z=2.4)
    const float3 lookat = make_float3(0.0f, 1.0f, 0.0f);   // 박스 중심 (Z=0)
    const float3 vup = make_float3(0.0f, 1.0f, 0.0f);
    const float fov = 45.0f; // 시야각
    const float aspect = (float)dim.x / (float)dim.y;

    // 간단한 카메라 로직 (w, u, v 기저 벡터)
    const float theta = fov * 3.14159265f / 180.0f;
    const float half_height = tanf(theta / 2.0f);
    const float half_width = aspect * half_height;
    const float3 w = normalize(lookfrom - lookat);
    const float3 u = normalize(cross(vup, w));
    const float3 v = cross(w, u);

    // 2. 초기 Ray 설정
    // 픽셀의 중심을 향해 쏘도록 설정
    float s = (idx.x + 0.5f) / (float)dim.x;
    float t = (idx.y + 0.5f) / (float)dim.y; // Y축 방향 주의 (필요시 1-t)

    float3 ray_origin = lookfrom;
    float3 ray_dir = normalize(u * (s - 0.5f) * half_width * 2.0f +
                               v * (0.5f - t) * half_height * 2.0f - w); // Y축 반전 처리

    // 3. Payload 초기화
    unsigned int seed = pixel_index + params.frame_index * 719393;

    RayPayload prd;
    prd.radiance = make_float3(0.0f);
    prd.throughput = make_float3(1.0f);
    prd.done = false;
    prd.seed = seed;

    unsigned int p0, p1;
    packPointer(&prd, p0, p1);

    // 4. Path Tracing Loop (Iterative)
    // 재귀 호출 대신 루프를 돕니다 (GPU 스택 오버플로우 방지)
    const int max_depth = 5;

    for (int depth = 0; depth < max_depth; ++depth)
    {
        prd.done = false; // Trace 전에 초기화

        optixTrace(
            params.handle,
            ray_origin,
            ray_dir,
            0.001f, 1e16f, 0.0f, // tmin, tmax, time
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0, 1, 0,
            p0, p1 // Payload 전달
        );

        // Miss 쉐이더나 Hit 쉐이더가 prd를 업데이트했음

        if (prd.done)
            break; // 빛이 소멸했거나 배경으로 나감

        // 다음 바운스를 위해 Ray 업데이트
        ray_origin = prd.origin;
        ray_dir = prd.direction;
    }

    float3 current_radiance = prd.radiance;

    // NaN(숫자가 아님) 체크 (가끔 튀는 픽셀 방지)
    if (isnan(current_radiance.x) || isnan(current_radiance.y) || isnan(current_radiance.z))
    {
        current_radiance = make_float3(0.0f);
    }

    // 2. Accumulation (누적)
    // 이전 프레임까지 누적된 값 읽어오기
    float4 prev_accum = make_float4(0.0f);
    if (params.frame_index > 0)
    {
        prev_accum = params.accum_buffer[pixel_index];
    }

    // 이번 값 더하기
    float4 new_accum = prev_accum + make_float4(current_radiance, 1.0f);
    params.accum_buffer[pixel_index] = new_accum;

    // 3. 평균 내기 (Averaging)
    float3 final_color = make_float3(new_accum) / new_accum.w; // w에는 샘플 수가 누적됨

    // 4. 감마 보정 및 출력
    // Clamp
    final_color.x = fminf(final_color.x, 1.0f);
    final_color.y = fminf(final_color.y, 1.0f);
    final_color.z = fminf(final_color.z, 1.0f);

    // Gamma 2.2
    params.image[pixel_index] = make_uchar4(
        (unsigned char)(powf(final_color.x, 1.0f / 2.2f) * 255.0f),
        (unsigned char)(powf(final_color.y, 1.0f / 2.2f) * 255.0f),
        (unsigned char)(powf(final_color.z, 1.0f / 2.2f) * 255.0f),
        255);
}

// Miss Program
extern "C" __global__ void __miss__ms()
{
    // Payload 가져오기
    unsigned int p0 = optixGetPayload_0();
    unsigned int p1 = optixGetPayload_1();
    RayPayload *prd = (RayPayload *)unpackPointer(p0, p1);

    // 배경색 (검은색)
    // prd->radiance += prd->throughput * make_float3(0.0f);

    prd->done = true; // 더 이상 추적 안 함
}

// Closest Hit Program
extern "C" __global__ void __closesthit__ch()
{
    // Payload 가져오기
    unsigned int p0 = optixGetPayload_0();
    unsigned int p1 = optixGetPayload_1();
    RayPayload *prd = (RayPayload *)unpackPointer(p0, p1);

    // 1. 교차 정보 가져오기
    const unsigned int prim_idx = optixGetPrimitiveIndex();
    const float t_hit = optixGetRayTmax();
    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();

    // 2. 지오메트리 법선(Normal) 계산
    // Triangle Mesh의 Vertex 정보가 필요함!
    const uint3 idx = params.indices[prim_idx];
    const float3 v0 = params.vertices[idx.x];
    const float3 v1 = params.vertices[idx.y];
    const float3 v2 = params.vertices[idx.z];

    // Face Normal (Flat Shading)
    // 외적 방향 주의 (CCW 기준)
    float3 N = normalize(cross(v1 - v0, v2 - v0));

    // 만약 Ray가 뒷면을 때렸으면 노말 뒤집기
    if (dot(ray_dir, N) > 0.0f)
        N = -N;

    // 교차점 P
    float3 P = ray_origin + t_hit * ray_dir;

    // 3. 재질(Material) 가져오기
    unsigned int mat_idx = params.matIndices[prim_idx];
    MaterialData mat = params.materials[mat_idx];

    // 4. 발광(Emission) 처리
    // Radiance += Throughput * Emission
    prd->radiance += prd->throughput * mat.emission;

    // 발광체라면 여기서 멈출 수도 있고, 반사시킬 수도 있음. 보통 멈추거나 아주 약하게 반사.
    // CornellBox 조명은 매우 밝으므로 Diffuse가 0이면 멈추는 게 효율적
    if (mat.diffuse.x < 1e-6f && mat.diffuse.y < 1e-6f && mat.diffuse.z < 1e-6f)
    {
        prd->done = true;
        return;
    }

    // 5. 다음 Ray 생성을 위한 샘플링 (Diffuse Reflection)
    // Cosine Weighted Hemisphere Sampling

    // 로컬 좌표계 생성
    float3 U, V;
    get_onb(N, U, V);

    // 랜덤 샘플링
    float r1 = rnd(prd->seed);
    float r2 = rnd(prd->seed);

    float3 local_dir;
    cosine_sample_hemisphere(r1, r2, local_dir);

    // 월드 좌표계로 변환
    float3 new_dir = local_dir.x * U + local_dir.y * V + local_dir.z * N;
    new_dir = normalize(new_dir);

    // 6. Throughput 업데이트 (색상 감쇠)
    // Rendering Equation: BRDF * CosTheta / PDF
    // Diffuse Lambertian:
    // BRDF = Color / PI
    // PDF = CosTheta / PI
    // Result = (Color / PI) * CosTheta / (CosTheta / PI) = Color
    // 즉, Throughput에 그냥 색상만 곱하면 됨 (Cosine Importance Sampling 덕분)

    prd->throughput *= mat.diffuse;

    // 7. 러시안 룰렛 (선택 사항: 깊이가 깊어지면 확률적으로 종료)
    // 일단 지금은 max_depth로 끊으므로 생략

    // 8. 다음 추적 준비
    prd->origin = P;
    prd->direction = new_dir;
}
