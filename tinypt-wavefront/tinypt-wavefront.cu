#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <iostream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <cmath>

#define PI 3.1415926535897932385
#define INV_PI 0.31830988618379067154

#include "vec3.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobjloader/tiny_obj_loader.h"

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
};

struct Ray
{
    Vec3 orig, dir;
    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const Vec3 &origin, const Vec3 &direction) : orig(origin), dir(direction) {}
    __host__ __device__ Vec3 at(float t) const { return orig + t * dir; }
};

struct RayPayload
{
    float t;       // t-value of intersection
    Vec3 p;        // intersected point
    Vec3 n;        // normal vector of intersected point
    uint32_t mIdx; // material index
};

struct HitQueue
{
    int triIdx;
    float t;
    float u, v;
};

struct RayWorkItem
{
    int pixelIndex;
    Vec3 origin;
    Vec3 direction;
};

struct PixelState
{
    Vec3 throughput;
    Vec3 accumulatedColor;
    int depth;
};

struct Triangle
{
    Vec3 v0, v1, v2, e1, e2, n;
    uint32_t matId;

    __host__ __device__ Triangle() {}
    __host__ __device__ Triangle(const Vec3 &v0, const Vec3 &v1, const Vec3 &v2, const uint32_t &materialId) : v0(v0), v1(v1), v2(v2), matId(materialId)
    {
        e1 = v1 - v0;
        e2 = v2 - v0;
        n = normalize(cross(e1, e2));
    }
    __host__ __device__ Vec3 normal() const { return n; }
    __host__ __device__ Vec3 edge1() const { return e1; }
    __host__ __device__ Vec3 edge2() const { return e2; }
    __host__ __device__ Vec3 centroid() const { return (v0 + v1 + v2) / 3.0f; }
    __host__ __device__ float area() const { return length(cross(e1, e2)) / 2.0f; }

    __device__ bool intersect(const Ray &r, float tMin, float tMax, RayPayload &payload) const
    {
        Vec3 v0v1 = v1 - v0;
        Vec3 v0v2 = v2 - v0;
        Vec3 pvec = cross(r.dir, v0v2);

        float det = dot(v0v1, pvec);

        if (fabs(det) < 1e-8f)
            return false;
        float invDet = 1.0f / det;

        Vec3 tvec = r.orig - v0;
        float u = dot(tvec, pvec) * invDet;
        if (u < 0.0f || u > 1.0f)
            return false;

        Vec3 qvec = cross(tvec, v0v1);
        float v = dot(r.dir, qvec) * invDet;
        if (v < 0.0f || u + v > 1.0f)
            return false;

        float t = dot(v0v2, qvec) * invDet;

        if (t < tMax && t > tMin)
        {
            payload.t = t;
            payload.p = r.at(t);
            payload.n = n; // 미리 계산된 법선 사용 (Flat Shading)
            payload.mIdx = matId;
            return true;
        }

        return false;
    }
    __device__ bool intersectP(const Ray &r, float tMin, float tMax, RayPayload &payload) const
    {
        Vec3 v0v1 = v1 - v0;
        Vec3 v0v2 = v2 - v0;
        Vec3 pvec = cross(r.dir, v0v2);

        float det = dot(v0v1, pvec);

        if (fabs(det) < 1e-8f)
            return false;
        float invDet = 1.0f / det;

        Vec3 tvec = r.orig - v0;
        float u = dot(tvec, pvec) * invDet;
        if (u < 0.0f || u > 1.0f)
            return false;

        Vec3 qvec = cross(tvec, v0v1);
        float v = dot(r.dir, qvec) * invDet;
        if (v < 0.0f || u + v > 1.0f)
            return false;

        float t = dot(v0v2, qvec) * invDet;

        if (t < tMax && t > tMin)
        {
            return true;
        }
        return false;
    }
};

enum MaterialType
{
    eLAMBERTIAN,
    eDIELECTRIC,
    eSPECULAR,
    eDIFFUSE_LIGHT,
};

struct Material
{
    MaterialType type;
    Vec3 albedo, emission;
    __host__ __device__ Material() : albedo(Vec3(0, 0, 0)), emission(Vec3(0, 0, 0)) {}
};

struct Scene
{
    std::vector<Triangle> triangles;
    std::vector<Material> materials;
};

Scene loadScene(const std::string &fp, const std::string &mtlFp)
{
    Scene scn;
    tinyobj::ObjReaderConfig cfg;
    cfg.mtl_search_path = mtlFp;
    cfg.triangulate = true;

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(fp, cfg))
    {
        if (!reader.Error().empty())
        {
            std::cerr << "TinyObjReader Error: " << reader.Error() << std::endl;
        }
        return scn;
    }

    if (!reader.Warning().empty())
    {
        std::cerr << "TinyObjReader Warning: " << reader.Warning() << std::endl;
    }

    auto &attrib = reader.GetAttrib();
    auto &shapes = reader.GetShapes();
    auto &materials = reader.GetMaterials();

    for (const auto &tMat : materials)
    {
        Material m;
        m.albedo = Vec3(tMat.diffuse[0], tMat.diffuse[1], tMat.diffuse[2]);
        m.emission = Vec3(tMat.emission[0], tMat.emission[1], tMat.emission[2]);

        // for checking light
        float emissionIntensity = m.emission.x + m.emission.y + m.emission.z;

        if (emissionIntensity > 1e-4f)
        {
            m.type = eDIFFUSE_LIGHT;
        }
        else if (tMat.illum == 5)
        {
            m.type = eSPECULAR;
        }
        else if (tMat.illum == 7)
        {
            m.type = eDIELECTRIC;
        }
        else
        {
            m.type = eLAMBERTIAN;
        }

        scn.materials.push_back(m);
    }

    if (scn.materials.empty())
    {
        std::cerr << "Materials of scene is empty." << std::endl;
    }

    for (const auto &shape : shapes)
    {
        size_t idxOffset = 0;
        for (size_t face = 0; face < shape.mesh.num_face_vertices.size(); face++)
        {
            int fv = shape.mesh.num_face_vertices[face];

            Vec3 vertices[3];
            for (int i = 0; i < 3; i++)
            {
                tinyobj::index_t idx = shape.mesh.indices[idxOffset + i];

                vertices[i] = Vec3(
                    attrib.vertices[3 * idx.vertex_index + 0],
                    attrib.vertices[3 * idx.vertex_index + 1],
                    attrib.vertices[3 * idx.vertex_index + 2]);
            }

            int matId = shape.mesh.material_ids[face];
            if (matId < 0 || matId >= scn.materials.size())
            {
                matId = 0;
            }

            scn.triangles.push_back(Triangle(vertices[0], vertices[1], vertices[2], matId));
            idxOffset += fv;
        }
    }
    return scn;
}

__global__ void initRandState(int width, int height, curandState *rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= width) || (j >= height))
        return;

    int pIdx = j * width + i;
    curand_init(1984 + pIdx, 0, 0, &rand_state[pIdx]);
}

__device__ Vec3 randomInUnitSphere(curandState *localRandState)
{
    Vec3 p;
    do
    {
        // -1.0 ~ 1.0 사이의 랜덤 값 추출
        float r1 = curand_uniform(localRandState);
        float r2 = curand_uniform(localRandState);
        float r3 = curand_uniform(localRandState);
        p = 2.0f * Vec3(r1, r2, r3) - Vec3(1.0f, 1.0f, 1.0f);
    } while (dot(p, p) >= 1.0f);
    return p;
}

__device__ float schlick(float cosine, float ref_idx)
{
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ bool refract(const Vec3 &v, const Vec3 &n, float ni_over_nt, Vec3 &refracted)
{
    Vec3 uv = normalize(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1.0f - dt * dt);
    if (discriminant > 0)
    {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }
    return false;
}

__device__ Vec3 reflect(const Vec3 &v, const Vec3 &n)
{
    return v - 2.0f * dot(v, n) * n;
}

// Wavefront path tracing
__global__ void generateRays(
    int width, int height,
    Camera cam, int spp_iter,
    RayWorkItem *rayQ,
    PixelState *stateQ,
    int *queueSize, curandState *randState)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height))
        return;

    int pIdx = j * width + i;
    curandState localRand = randState[pIdx];

    float u = (float(i) + curand_uniform(&localRand)) / float(width);
    float v = (float(j) + curand_uniform(&localRand)) / float(height);

    Vec3 dir = normalize(cam.lowerLeftCorner + u * cam.horizontal + v * cam.vertical - cam.origin);
    rayQ[pIdx].origin = cam.origin;
    rayQ[pIdx].direction = dir;
    rayQ[pIdx].pixelIndex = pIdx;

    stateQ[pIdx].throughput = Vec3(1.0f);
    stateQ[pIdx].accumulatedColor = Vec3(0.0f);
    stateQ[pIdx].depth = 0;

    randState[pIdx] = localRand;

    if (pIdx == 0)
        *queueSize = width * height;
}

__global__ void extend(
    int queueSize,
    RayWorkItem *rayQ, HitQueue *hitQ,
    Triangle *triangles, int numTriangles)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= queueSize)
        return;

    RayWorkItem r = rayQ[idx];
    Ray ray(r.origin, r.direction);

    RayPayload payload;
    float closest = 1e20f;
    int hitIdx = -1;

    RayPayload tmp;
    for (int i = 0; i < numTriangles; i++)
    {
        if (triangles[i].intersect(ray, 0.001f, closest, tmp))
        {
            closest = tmp.t;
            hitIdx = i;
            payload = tmp;
        }
    }

    hitQ[idx].triIdx = hitIdx;
    hitQ[idx].t = closest;
}

__global__ void shadeAndEnqueue(
    int numCurrentRays,
    int *d_nextQueueCount,
    const RayWorkItem *currentQ,
    const HitQueue *hitQ,
    RayWorkItem *nextQ,
    PixelState *pixelStates,
    Vec3 *frameBuffer,
    Triangle *triangles,
    Material *materials,
    curandState *randState,
    int maxDepth)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numCurrentRays)
        return;

    RayWorkItem rItem = currentQ[idx];
    HitQueue hit = hitQ[idx];
    int pIdx = rItem.pixelIndex;

    PixelState &state = pixelStates[pIdx];

    // 1. Miss 처리
    if (hit.triIdx == -1)
    {
        frameBuffer[pIdx] += state.accumulatedColor;
        // active 플래그 대신 큐에 넣지 않음으로써 종료
        return;
    }

    Triangle tri = triangles[hit.triIdx];
    Material mat = materials[tri.matId];

    Vec3 hitPoint = rItem.origin + rItem.direction * hit.t;
    Vec3 normal = tri.normal();
    bool front_face = dot(rItem.direction, normal) < 0;
    Vec3 outward_normal = front_face ? normal : -normal; // 항상 밖을 향하는 법선

    // 2. Emission (Le) 더하기
    if (mat.emission.x > 0 || mat.emission.y > 0 || mat.emission.z > 0)
    {
        state.accumulatedColor += state.throughput * mat.emission;
    }

    // 3. 종료 조건 확인
    if (mat.type == eDIFFUSE_LIGHT || state.depth >= maxDepth)
    {
        frameBuffer[pIdx] += state.accumulatedColor;
        return;
    }

    // 4. 재질별 BSDF 샘플링
    curandState localRand = randState[pIdx];
    Vec3 nextDir;
    float pdf = 0.0f; // 확률 밀도 함수 (설명을 위해 변수만 둠)

    if (mat.type == eLAMBERTIAN)
    {
        Vec3 target = outward_normal + randomInUnitSphere(&localRand);
        if (dot(target, target) < 1e-8f)
            target = outward_normal;
        nextDir = normalize(target);

        // Lambertian: f = albedo/PI, pdf = cos/PI -> weight = albedo
        state.throughput = state.throughput * mat.albedo;
    }
    else if (mat.type == eSPECULAR)
    {
        nextDir = reflect(normalize(rItem.direction), outward_normal);
        // Specular: 완벽한 반사이므로 감쇠만 적용
        state.throughput = state.throughput * mat.albedo;
    }
    else if (mat.type == eDIELECTRIC)
    {
        Vec3 outward_normal_dielectric;
        Vec3 reflected = reflect(rItem.direction, normal);
        float ni_over_nt;
        float reflect_prob;
        float cosine;

        // 유리 굴절률 (공기 1.0, 유리 1.5)
        if (front_face)
        {
            outward_normal_dielectric = normal;
            ni_over_nt = 1.0f / 1.5f;
            cosine = -dot(rItem.direction, normal);
        }
        else
        {
            outward_normal_dielectric = -normal;
            ni_over_nt = 1.5f;
            cosine = 1.5f * dot(rItem.direction, normal); // Snell's law 보정
        }

        Vec3 refracted;
        if (refract(rItem.direction, outward_normal_dielectric, ni_over_nt, refracted))
        {
            reflect_prob = schlick(cosine, 1.5f);
        }
        else
        {
            reflect_prob = 1.0f; // 전반사 (Total Internal Reflection)
        }

        if (curand_uniform(&localRand) < reflect_prob)
        {
            nextDir = reflected;
        }
        else
        {
            nextDir = refracted;
        }

        // Dielectric은 빛을 흡수하지 않음 (albedo가 보통 1.0)
        state.throughput = state.throughput * mat.albedo;
    }

    // 상태 업데이트
    state.depth++;
    randState[pIdx] = localRand;

    // 5. 다음 큐에 넣기 (Enqueue)
    int slot = atomicAdd(d_nextQueueCount, 1);

    RayWorkItem nextItem;
    nextItem.pixelIndex = pIdx;
    // Shadow Acne 방지: 법선 방향으로 아주 살짝 띄움
    // (주의: Dielectric은 굴절 시 안쪽으로 들어가야 하므로 offset 방향 주의해야 함.
    //  여기선 간단히 진행 방향으로 약간 띄우는 방식을 사용)
    nextItem.origin = hitPoint + nextDir * 0.001f;
    nextItem.direction = nextDir;

    nextQ[slot] = nextItem;
}

// Megakernel path tracing
__device__ bool traceRay(
    const Ray &r, RayPayload &payload,
    const Triangle *triangles, int numTriangles,
    const Material *materials, int numMaterials,
    float tMin, float tMax)
{
    RayPayload tmpPayload;
    bool anyHit = false;
    float closestSoFar = tMax;

    for (int i = 0; i < numTriangles; i++)
    {
        if (triangles[i].intersect(r, tMin, closestSoFar, tmpPayload))
        {
            anyHit = true;
            closestSoFar = tmpPayload.t;
            payload = tmpPayload;
        }
    }

    return anyHit;
}

__device__ Vec3 radiance(
    Ray &r, int maxDepth,
    const Triangle *triangles, int numTriangles,
    const Material *materials, int numMaterials,
    curandState *randState)
{
    Vec3 color(0.0f);
    Vec3 thp(1.0f);
    Ray ray = r;

    for (int depth = 0; depth < maxDepth; depth++)
    {
        RayPayload payload;
        if (traceRay(ray, payload, triangles, numTriangles, materials, numMaterials, 0.00001f, 1e20f))
        {
            const Material &mat = materials[payload.mIdx];

            // L = L + Le
            color += thp * mat.emission;

            if (mat.type == eDIFFUSE_LIGHT)
            {
                break;
            }

            Vec3 target = payload.n + randomInUnitSphere(randState);
            if (dot(target, target) < 1e-8f)
            {
                target = payload.n;
            }

            Ray outRay(payload.p, normalize(target));

            Vec3 attenuation = mat.albedo;
            thp = thp * attenuation;
            ray = outRay;
        }
        else
        {
            break;
        }
    }
    return color;
}

__global__ void render(
    Vec3 *fb,
    int width, int height, int spp,
    Camera cam,
    Triangle *triangles, int numTriangles,
    Material *materials, int numMaterials,
    curandState *randState)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= width) || (j >= height))
        return;

    int pIdx = j * width + i;
    curandState localRandState = randState[pIdx];

    Vec3 color(0.0f);

    for (int s = 0; s < spp; s++)
    {
        float u = (float(i) + curand_uniform(&localRandState)) / float(width);
        float v = (float(j) + curand_uniform(&localRandState)) / float(height);

        Ray ray = Ray(cam.origin, normalize(cam.lowerLeftCorner + u * cam.horizontal + v * cam.vertical - cam.origin));
        color += radiance(ray, 5, triangles, numTriangles, materials, numMaterials, &localRandState);
    }

    color /= float(spp);
    fb[pIdx] = color;
    randState[pIdx] = localRandState;
}

int main(int argc, char *argv[])
{
    // 1. Scene & Resource Loading (기존과 동일)
    std::string inputPath = (argc > 1) ? argv[1] : "scenes/CornellBox/CornellBox-Original.obj";
    std::filesystem::path objPath(inputPath);

    if (!std::filesystem::exists(objPath))
    {
        std::cerr << "[Error]: File not found - " << inputPath << std::endl;
        return -1;
    }

    std::string objFp = objPath.string();
    std::string mtlDir = objPath.parent_path().string() + "/";

    std::cout << "[Info] OBJ Path: " << objFp << std::endl;
    std::cout << "[Info] MTL Dir : " << mtlDir << std::endl;

    int width = 1000;
    int height = 1000;
    int numPixels = width * height; // 편의를 위한 변수
    float aspectRatio = float(width) / float(height);
    
    Scene scn = loadScene(objFp, mtlDir);

    // Camera Setup (Cornell Box에 맞춰 조정됨)
    Camera cam(Vec3(0.0f, 1.0f, 4.0f), Vec3(0.0f, 1.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), 40.0f, aspectRatio);

    // 2. Memory Allocation for Scene Data
    int numTriangles = scn.triangles.size();
    int numMaterials = scn.materials.size();

    Triangle *dTriangles;
    Material *dMaterials;
    Vec3 *dFrameBuffer;
    curandState *dRandState;

    cudaMalloc((void **)&dTriangles, numTriangles * sizeof(Triangle));
    cudaMalloc((void **)&dMaterials, numMaterials * sizeof(Material));
    cudaMalloc((void **)&dFrameBuffer, numPixels * sizeof(Vec3));
    cudaMalloc((void **)&dRandState, numPixels * sizeof(curandState));

    // Data Upload
    cudaMemcpy(dTriangles, scn.triangles.data(), numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMemcpy(dMaterials, scn.materials.data(), numMaterials * sizeof(Material), cudaMemcpyHostToDevice);
    cudaMemset(dFrameBuffer, 0, numPixels * sizeof(Vec3)); // 초기화 필수

    // 3. [Wavefront] Memory Allocation for Queues & States
    RayWorkItem *d_queueCurrent, *d_queueNext; // Ping-Pong Queues
    HitQueue *d_hitQueue;
    PixelState *d_pixelStates;
    int *d_numRays;      // 현재 큐에 담긴 레이 개수 (Device 변수)
    int *d_nextNumRays;  // 다음 큐에 담길 레이 개수 (Device 변수 - Atomic용)

    cudaMalloc((void **)&d_queueCurrent, numPixels * sizeof(RayWorkItem));
    cudaMalloc((void **)&d_queueNext,    numPixels * sizeof(RayWorkItem));
    cudaMalloc((void **)&d_hitQueue,     numPixels * sizeof(HitQueue));
    cudaMalloc((void **)&d_pixelStates,  numPixels * sizeof(PixelState));
    cudaMalloc((void **)&d_numRays,      sizeof(int));
    cudaMalloc((void **)&d_nextNumRays,  sizeof(int));

    // 4. Random State Initialization
    int tx = 8, ty = 8;
    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);

    initRandState<<<blocks, threads>>>(width, height, dRandState);
    cudaDeviceSynchronize();

    std::cout << "[Info] Starting Wavefront Path Tracing..." << std::endl;

// ----------------------------------------------------------------
    // 5. Wavefront Main Loop
    // ----------------------------------------------------------------
    int spp = 4096;      // 샘플 수 (테스트 시 조절하세요)
    int maxDepth = 5;   // 최대 반사 횟수

    for (int s = 0; s < spp; s++)
    {
        // [Step 1] Generate Rays (Primary Rays)
        // 초기화: d_numRays를 width * height로 설정하고, 모든 픽셀에 대해 레이 생성
        generateRays<<<blocks, threads>>>(
            width, height, cam, s, 
            d_queueCurrent, d_pixelStates, d_numRays, dRandState
        );
        cudaDeviceSynchronize(); // Generate 완료 대기

        int h_numRays = 0; // 호스트에서 현재 레이 개수 추적용

        for (int depth = 0; depth < maxDepth; depth++)
        {
            // Device에 있는 레이 개수를 Host로 가져옴
            cudaMemcpy(&h_numRays, d_numRays, sizeof(int), cudaMemcpyDeviceToHost);
            
            if (h_numRays == 0) break; // 처리할 레이가 없으면 조기 종료

            // 다음 단계 카운터(Atomic용) 0으로 초기화
            cudaMemset(d_nextNumRays, 0, sizeof(int));

            // Grid 크기 계산 (1D Grid)
            int numBlocks = (h_numRays + 255) / 256;

            // [Step 2] Extend (Intersection Test)
            // 현재 큐(d_queueCurrent)에 있는 h_numRays 만큼만 수행
            extend<<<numBlocks, 256>>>(
                h_numRays, 
                d_queueCurrent, d_hitQueue, 
                dTriangles, numTriangles
            );
            cudaDeviceSynchronize(); // 디버깅 시 필요할 수 있음

            // [Step 3] Shade & Enqueue (Logic)
            // 교차 결과를 바탕으로 쉐이딩하고, 살아남은 레이를 d_queueNext에 넣음
            shadeAndEnqueue<<<numBlocks, 256>>>(
                h_numRays, 
                d_nextNumRays, 
                d_queueCurrent, d_hitQueue, 
                d_queueNext, 
                d_pixelStates, dFrameBuffer, 
                dTriangles, dMaterials, dRandState, 
                maxDepth
            );
            cudaDeviceSynchronize(); 

            // [Step 4] Ping-Pong (Swap Queues)
            // 큐 포인터 교체 (다음 루프에서는 next가 current가 됨)
            std::swap(d_queueCurrent, d_queueNext);
            
            // 레이 개수 업데이트: d_nextNumRays(계산된 값) -> d_numRays(다음 루프 입력)
            cudaMemcpy(d_numRays, d_nextNumRays, sizeof(int), cudaMemcpyDeviceToDevice);
        }

        // 진행 상황 표시 (10번째 샘플마다)
        if ((s + 1) % 10 == 0) 
            std::cout << "Sample: " << (s + 1) << " / " << spp << "\r" << std::flush;
    }

    std::cout << std::endl << "[Info] Rendering Done." << std::endl;

    // 6. Image Saving & Post-Processing
    std::cout << "[Info] Processing and saving image..." << std::endl;
    
    std::vector<Vec3> fb(numPixels);
    cudaMemcpy(fb.data(), dFrameBuffer, numPixels * sizeof(Vec3), cudaMemcpyDeviceToHost);

    std::vector<unsigned char> image(numPixels * 3);

    for (int j = 0; j < height; ++j)
    {
        for (int i = 0; i < width; ++i)
        {
            size_t pixel_index = (height - 1 - j) * width + i;
            Vec3 pixel = fb[pixel_index];

            // [중요] Wavefront에서는 accumulate만 했으므로 여기서 SPP로 나누어 줍니다.
            pixel /= float(spp);

            float r = sqrt(pixel.x); // Gamma Correction
            float g = sqrt(pixel.y);
            float b = sqrt(pixel.z);

            int ir = int(255.99f * (r > 1.0f ? 1.0f : (r < 0.0f ? 0.0f : r)));
            int ig = int(255.99f * (g > 1.0f ? 1.0f : (g < 0.0f ? 0.0f : g)));
            int ib = int(255.99f * (b > 1.0f ? 1.0f : (b < 0.0f ? 0.0f : b)));

            size_t img_index = (j * width + i) * 3;
            image[img_index + 0] = static_cast<unsigned char>(ir);
            image[img_index + 1] = static_cast<unsigned char>(ig);
            image[img_index + 2] = static_cast<unsigned char>(ib);
        }
    }

    if (stbi_write_png("tinypt_wavefront.png", width, height, 3, image.data(), width * 3))
    {
        std::cout << "[Info] Image saved successfully: tinypt_wavefront.png" << std::endl;
    }
    else
    {
        std::cerr << "[Error] Failed to save image!" << std::endl;
    }

    // 7. Cleanup
    cudaFree(dTriangles);
    cudaFree(dMaterials);
    cudaFree(dFrameBuffer);
    cudaFree(dRandState);
    
    // Wavefront Buffers Free
    cudaFree(d_queueCurrent);
    cudaFree(d_queueNext);
    cudaFree(d_hitQueue);
    cudaFree(d_pixelStates);
    cudaFree(d_numRays);
    cudaFree(d_nextNumRays);

    return 0;
}