#include <cuda_runtime.h>
#include <curand_kernel.h>

// for lbvh
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include <iostream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stack>

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

/*
    Linear Bounding Volume Hierarchy
*/

struct AABB
{
    Vec3 min, max;

    __host__ __device__ AABB() : min(Vec3(1e30f)), max(Vec3(-1e30f)) {}
    __host__ __device__ AABB(const Vec3 &min, const Vec3 &max) : min(min), max(max) {}

    __host__ __device__ void grow(const Vec3 &p)
    {
        min.x = fminf(min.x, p.x);
        min.y = fminf(min.y, p.y);
        min.z = fminf(min.z, p.z);

        max.x = fmaxf(max.x, p.x);
        max.y = fmaxf(max.y, p.y);
        max.z = fmaxf(max.z, p.z);
    }

    __host__ __device__ void grow(const AABB &aabb)
    {
        min.x = fminf(min.x, aabb.min.x);
        min.y = fminf(min.y, aabb.min.y);
        min.z = fminf(min.z, aabb.min.z);

        max.x = fmaxf(max.x, aabb.max.x);
        max.y = fmaxf(max.y, aabb.max.y);
        max.z = fmaxf(max.z, aabb.max.z);
    }

    __host__ __device__ void grow(const Triangle &tri)
    {
        grow(tri.v0);
        grow(tri.v1);
        grow(tri.v2);
    }
};

__device__ uint32_t expandBits(uint32_t v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ uint32_t morton3D(float x, float y, float z)
{
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);

    uint32_t xx = expandBits((uint32_t)x);
    uint32_t yy = expandBits((uint32_t)y);
    uint32_t zz = expandBits((uint32_t)z);

    return xx * 4 + yy * 2 + zz;
}

__global__ void computeMortonCodes(
    const Triangle *triangles,
    uint32_t *mortonCodes,
    uint32_t *objectIds,
    int numTriangles,
    AABB sceneBounds)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTriangles)
        return;

    // calculate centroid of triangle
    Vec3 centroid = triangles[idx].centroid();

    // calculation and nomalization extent vector according to scene bounds
    Vec3 extent = sceneBounds.max - sceneBounds.min;
    Vec3 normalizedPos = (centroid - sceneBounds.min) / extent;

    mortonCodes[idx] = morton3D(normalizedPos.x, normalizedPos.y, normalizedPos.z);
    objectIds[idx] = idx;
}

void sortMortonCodes(
    uint32_t *dMortonCodes,
    uint32_t *dObjectIds,
    int numTriangles)
{
    // wrapping pointer
    thrust::device_ptr<uint32_t> tMortonCodes(dMortonCodes);
    thrust::device_ptr<uint32_t> tObjectIds(dObjectIds);

    // sort by key by using radix sort
    thrust::sort_by_key(tMortonCodes, tMortonCodes + numTriangles, tObjectIds);
}

struct LbvhNode
{
    AABB bounds;
    int left, right;
    // if index < 0: it is leaf node
    __host__ __device__ LbvhNode() : left(-1), right(-1) {}
};

__device__ int longestCommonPrefix(int i, int j, int numTriangles, const uint32_t *mortonCodes)
{
    if (i < 0 || i >= numTriangles || j < 0 || j >= numTriangles)
        return -1;

    uint32_t code1 = mortonCodes[i];
    uint32_t code2 = mortonCodes[j];

    if (code1 == code2)
    {
        return __clz(i ^ j) + 32;
    }

    return __clz(code1 ^ code2);
}

__global__ void generateHierarchy(
    LbvhNode *nodes,
    const uint32_t *mortonCodes,
    int *parents,
    int numTriangles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTriangles - 1)
        return;

    // 1. direction define
    int deltaNext = longestCommonPrefix(idx, idx + 1, numTriangles, mortonCodes);
    int deltaPrev = longestCommonPrefix(idx, idx - 1, numTriangles, mortonCodes);

    int direction = (deltaNext > deltaPrev) ? 1 : -1;
    int deltaMin = longestCommonPrefix(idx, idx - direction, numTriangles, mortonCodes);

    // 2. finding last effect range
    int lMax = 2;
    while (longestCommonPrefix(idx, idx + lMax * direction, numTriangles, mortonCodes) > deltaMin)
    {
        lMax *= 2;
    }

    int l = 0;
    for (int t = lMax / 2; t >= 1; t /= 2)
    {
        if (longestCommonPrefix(idx, idx + (l + t) * direction, numTriangles, mortonCodes) > deltaMin)
        {
            l += t;
        }
    }

    int j = idx + l * direction;

    // 3. finding split position
    int deltaNode = longestCommonPrefix(idx, j, numTriangles, mortonCodes);
    int s = 0;

    int first = min(idx, j);
    int last = max(idx, j);

    if (deltaMin < deltaNode)
    {
        int t = l;
        do
        {
            t = (t + 1) / 2;
            if (longestCommonPrefix(first, first + (s + t), numTriangles, mortonCodes) > deltaNode)
            {
                s += t;
            }
        } while (t > 1);
    }
    int split = first + s;

    // 4. connection child node
    if (min(idx, j) == split)
    {
        int childIdx = split + (numTriangles - 1);
        nodes[idx].left = childIdx; // leaf
        parents[childIdx] = idx;
    }
    else
    {
        int childIdx = split;
        nodes[idx].left = childIdx;
        parents[childIdx] = idx; // [중요]
    }

    if (max(idx, j) == split + 1)
    {
        // Leaf인 경우
        int childIdx = split + 1 + (numTriangles - 1);
        nodes[idx].right = childIdx;
        parents[childIdx] = idx; // [중요]
    }
    else
    {
        // Internal Node인 경우
        int childIdx = split + 1;
        nodes[idx].right = childIdx;
        parents[childIdx] = idx; // [중요]
    }
}

__global__ void refitBounds(
    int numTriangles,
    const uint32_t *objectIds,
    const Triangle *triangles,
    LbvhNode *nodes,
    const int *parents,
    int *flags)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTriangles)
        return;

    int triangleId = objectIds[idx];
    Triangle tri = triangles[triangleId];

    AABB currentBox;
    currentBox.grow(tri);

    int myIdx = (numTriangles - 1) + idx;
    nodes[myIdx].bounds = currentBox;
    int currentNodeIdx = myIdx;

    while (true)
    {
        int parentIdx = parents[currentNodeIdx];
        if (parentIdx == -1)
            break;

        int oldVal = atomicCAS(&flags[parentIdx], 0, 1);
        if (oldVal == 0)
        {
            // nodes[parentIdx].bounds = currentBox;
            __threadfence();
            break;
        }
        else
        {
            __threadfence(); // 형제의 데이터를 읽기 전 안전장치

            int leftChild = nodes[parentIdx].left;
            int rightChild = nodes[parentIdx].right;
            
            AABB boxLeft = nodes[leftChild].bounds;
            AABB boxRight = nodes[rightChild].bounds;

            // 두 박스를 합침
            AABB unionBox;
            unionBox.grow(boxLeft);
            unionBox.grow(boxRight);

            // 부모 노드에 최종 기록
            nodes[parentIdx].bounds = unionBox;

            // 이제 내가 부모를 대표해서 위로 올라감
            currentNodeIdx = parentIdx;
        }
    }
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

__device__ bool intersectAABB(const AABB& box, const Ray& r, float tMin, float tMax) {
    // 0으로 나누는 것을 방지하기 위해 아주 작은 값 더함 (혹은 r.dir이 0이 아니라고 가정)
    Vec3 invDir(1.0f / (r.dir.x + 1e-16f), 1.0f / (r.dir.y + 1e-16f), 1.0f / (r.dir.z + 1e-16f));

    Vec3 t0 = (box.min - r.orig) * invDir;
    Vec3 t1 = (box.max - r.orig) * invDir;

    Vec3 tSmall, tBig;
    tSmall.x = fminf(t0.x, t1.x); 
    tSmall.y = fminf(t0.y, t1.y); 
    tSmall.z = fminf(t0.z, t1.z);

    tBig.x = fmaxf(t0.x, t1.x);
    tBig.y = fmaxf(t0.y, t1.y);
    tBig.z = fmaxf(t0.z, t1.z);

    float tmin = fmaxf(tSmall.x, fmaxf(tSmall.y, fmaxf(tSmall.z, tMin)));
    float tmax = fminf(tBig.x, fminf(tBig.y, fminf(tBig.z, tMax)));

    return tmin <= tmax;
}

__device__ bool traceRay(
    const Ray &r, RayPayload &payload,
    const Triangle *triangles, int numTriangles,
    const Material *materials, int numMaterials,
    const LbvhNode *nodes, const uint32_t *objectIds,
    float tMin, float tMax)
{
    bool hit = false;
    float closestSoFar = tMax;

    // 스택 기반 순회 (Stack-based Traversal)
    // GPU 로컬 메모리에 스택 생성 (깊이 64면 충분)
    int stack[64];
    int stackPtr = 0;
    
    // 루트 노드(0번) 푸시
    stack[stackPtr++] = 0; 

    while (stackPtr > 0)
    {
        int nodeIdx = stack[--stackPtr]; // 팝

        // 리프 노드인지 확인 (우리는 idx >= numTriangles - 1 방식 사용 중)
        bool isLeaf = nodeIdx >= (numTriangles - 1);
        int actualArrIdx = isLeaf ? (nodeIdx - (numTriangles - 1)) : nodeIdx;

        if (isLeaf)
        {
            // 리프 노드: 실제 삼각형과 교차 검사
            int triIdx = objectIds[actualArrIdx]; // 정렬된 ID 테이블 참조
            RayPayload tmpPayload;
            
            // 기존 intersect 함수 사용
            if (triangles[triIdx].intersect(r, tMin, closestSoFar, tmpPayload))
            {
                hit = true;
                closestSoFar = tmpPayload.t;
                payload = tmpPayload;
            }
        }
        else
        {
            // 내부 노드: AABB 교차 검사
            const LbvhNode& node = nodes[actualArrIdx];
            
            // 현재 노드의 박스와 부딪히는지 검사
            if (intersectAABB(node.bounds, r, tMin, closestSoFar))
            {
                // 부딪혔으면 자식들을 스택에 추가
                // (가까운 것을 나중에 넣어서 먼저 꺼내면 최적화되지만, 일단 단순하게 넣음)
                stack[stackPtr++] = node.left;
                stack[stackPtr++] = node.right;
            }
        }
    }

    return hit;
}

__device__ Vec3 radiance(
    Ray &r, int maxDepth,
    const Triangle *triangles, int numTriangles,
    const Material *materials, int numMaterials,
    const LbvhNode *nodes, const uint32_t *objectIds,
    curandState *randState)
{
    Vec3 color(0.0f);
    Vec3 thp(1.0f);
    Ray ray = r;

    for (int depth = 0; depth < maxDepth; depth++)
    {
        RayPayload payload;
        if (traceRay(ray, payload, triangles, numTriangles, materials, numMaterials, nodes, objectIds, 0.00001f, 1e20f))
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
    LbvhNode* nodes, uint32_t* objectIds,
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
        color += radiance(ray, 5, triangles, numTriangles, materials, numMaterials, nodes, objectIds, &localRandState);
    }

    color /= float(spp);
    fb[pIdx] = color;
    randState[pIdx] = localRandState;
}

void checkLBVH(
    const std::vector<LbvhNode> &nodes,
    const std::vector<uint32_t> &objectIds, // 정렬된 리프 인덱스
    const std::vector<int> &parents,
    int numTriangles)
{
    std::cout << "\n[Debug] Starting LBVH Validation..." << std::endl;

    // 1. 루트 노드 확인 (0번 노드가 루트여야 함)
    // 루트의 부모는 -1이어야 함 (하지만 parents 배열은 자식->부모 매핑이므로 직접 확인 어려움)
    // 대신 루트의 AABB가 유효한지 확인
    std::cout << "  Root Bounds: "
              << nodes[0].bounds.min.x << " " << nodes[0].bounds.min.y << " " << nodes[0].bounds.min.z << " -> "
              << nodes[0].bounds.max.x << " " << nodes[0].bounds.max.y << " " << nodes[0].bounds.max.z << std::endl;

    if (nodes[0].bounds.min.x > nodes[0].bounds.max.x)
    {
        std::cerr << "  [Error] Root AABB is invalid! (Min > Max)" << std::endl;
    }

    // 2. 트리 순회하며 연결성 확인 (DFS)
    std::stack<int> stack;
    stack.push(0); // Root
    int visitedNodes = 0;
    int visitedLeaves = 0;

    while (!stack.empty())
    {
        int idx = stack.top();
        stack.pop();
        visitedNodes++;

        const LbvhNode &node = nodes[idx];

        // 자식 확인
        int left = node.left;
        int right = node.right;

        // 왼쪽 자식 검사
        if (left >= numTriangles - 1)
        { // Leaf Node
            visitedLeaves++;
            // 리프 노드의 인덱스가 유효한지 확인
            int leafIdx = left - (numTriangles - 1);
            if (leafIdx < 0 || leafIdx >= numTriangles)
                std::cerr << "  [Error] Invalid Left Leaf Index: " << leafIdx << " at Node " << idx << std::endl;
        }
        else
        { // Internal Node
            if (left < 0)
                std::cerr << "  [Error] Invalid Left Child: " << left << " at Node " << idx << std::endl;
            else
                stack.push(left);
        }

        // 오른쪽 자식 검사
        if (right >= numTriangles - 1)
        { // Leaf Node
            visitedLeaves++;
            int leafIdx = right - (numTriangles - 1);
            if (leafIdx < 0 || leafIdx >= numTriangles)
                std::cerr << "  [Error] Invalid Right Leaf Index: " << leafIdx << " at Node " << idx << std::endl;
        }
        else
        { // Internal Node
            if (right < 0)
                std::cerr << "  [Error] Invalid Right Child: " << right << " at Node " << idx << std::endl;
            else
                stack.push(right);
        }
    }

    std::cout << "  Visited Internal Nodes: " << visitedNodes << " (Expected: " << numTriangles - 1 << ")" << std::endl;
    std::cout << "  Visited Leaf Nodes: " << visitedLeaves << " (Expected: " << numTriangles << ")" << std::endl;

    if (visitedNodes == numTriangles - 1 && visitedLeaves == numTriangles)
    {
        std::cout << "  [Success] LBVH Topology seems correct!" << std::endl;
    }
    else
    {
        std::cerr << "  [Fail] Node counts do not match!" << std::endl;
    }
}

int main(int argc, char *argv[])
{
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

    // scene info
    int width = 1000;
    int height = 1000;
    float aspectRatio = float(width) / float(height);
    Scene scn = loadScene(objFp, mtlDir);

    // create camera
    // lookFrom, lookAt, vup, vfov, aspectRatio
    Camera cam(Vec3(0.0f, 1.0f, 4.0f), Vec3(0.0f, 1.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), 40.0f, aspectRatio);

    // calculation of memory for GPU
    int numTriangles = scn.triangles.size();
    int numMaterials = scn.materials.size();
    size_t triangleMemSize = numTriangles * sizeof(Triangle);
    size_t materialMemSize = numMaterials * sizeof(Material);
    size_t fbMemSize = width * height * sizeof(Vec3);
    size_t randStateSize = width * height * sizeof(curandState);

    Triangle *dTriangles;
    Material *dMaterials;
    Vec3 *dFrameBuffer;
    curandState *dRandState;

    cudaMalloc((void **)&dTriangles, triangleMemSize);
    cudaMalloc((void **)&dMaterials, materialMemSize);
    cudaMalloc((void **)&dFrameBuffer, fbMemSize);
    cudaMalloc((void **)&dRandState, randStateSize);

    cudaMemcpy(dTriangles, scn.triangles.data(), triangleMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dMaterials, scn.materials.data(), materialMemSize, cudaMemcpyHostToDevice);

    // build lbvh
    uint32_t *dMortonCodes, *dObjectIds;
    cudaMalloc(&dMortonCodes, numTriangles * sizeof(uint32_t));
    cudaMalloc(&dObjectIds, numTriangles * sizeof(uint32_t));

    LbvhNode *dNodes;
    int *dParents, *dFlags;
    cudaMalloc(&dNodes, 2 * numTriangles * sizeof(LbvhNode));
    cudaMalloc(&dParents, 2 * numTriangles * sizeof(int)); // 자식 수(2N)만큼 넉넉히
    cudaMalloc(&dFlags, (numTriangles - 1) * sizeof(int));
    cudaMemset(dFlags, 0, (numTriangles - 1) * sizeof(int)); // 0 초기화 필수

    std::cout << "[Info] Building LBVH..." << std::endl;
    
    AABB globalBounds;
    for (const auto& tri : scn.triangles) globalBounds.grow(tri); // CPU에서 계산 (간단하게)
    
    int blockSize = 256;
    int gridSize = (numTriangles + blockSize - 1) / blockSize;

    computeMortonCodes<<<gridSize, blockSize>>>(dTriangles, dMortonCodes, dObjectIds, numTriangles, globalBounds);
    cudaDeviceSynchronize();

    // -------------------------------------------------------------------------------------
    // [LBVH Step 2] Sort
    // -------------------------------------------------------------------------------------
    sortMortonCodes(dMortonCodes, dObjectIds, numTriangles);
    
    // -------------------------------------------------------------------------------------
    // [LBVH Step 3] Generate Hierarchy (Topology)
    // -------------------------------------------------------------------------------------
    // *주의: parents 배열 초기화 (-1)
    cudaMemset(dParents, -1, 2 * numTriangles * sizeof(int));
    
    generateHierarchy<<<gridSize, blockSize>>>(dNodes, dMortonCodes, dParents, numTriangles);
    cudaDeviceSynchronize();

    // -------------------------------------------------------------------------------------
    // [LBVH Step 4] Refit Bounds (AABB Calculation)
    // -------------------------------------------------------------------------------------
    refitBounds<<<gridSize, blockSize>>>(numTriangles, dObjectIds, dTriangles, dNodes, dParents, dFlags);
    cudaDeviceSynchronize();
    
    std::cout << "[Info] LBVH Build Finished." << std::endl;

    // GPU 데이터를 CPU로 복사해 와서 확인
    std::vector<LbvhNode> hNodes(numTriangles - 1);
    std::vector<uint32_t> hObjectIds(numTriangles);
    std::vector<int> hParents(2 * numTriangles);

    cudaMemcpy(hNodes.data(), dNodes, (numTriangles - 1) * sizeof(LbvhNode), cudaMemcpyDeviceToHost);
    cudaMemcpy(hObjectIds.data(), dObjectIds, numTriangles * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(hParents.data(), dParents, 2 * numTriangles * sizeof(int), cudaMemcpyDeviceToHost);

    // 검증 함수 호출
    checkLBVH(hNodes, hObjectIds, hParents, numTriangles);

    // ... (이후 렌더링 코드: dNodes, dObjectIds를 render 커널에 넘겨서 사용) ...
    
    // (메모리 해제는 프로그램 끝날 때 수행)
    cudaFree(dMortonCodes); 
    // dObjectIds와 dNodes는 렌더링 내내 써야 하므로 해제하면 안 됨!
    cudaFree(dParents); // 구축 끝나면 부모 정보는 필요 없음 (선택)
    cudaFree(dFlags);   // 구축 끝나면 필요 없음


    int tx = 8, ty = 8;
    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);

    initRandState<<<blocks, threads>>>(width, height, dRandState);
    cudaDeviceSynchronize();

    render<<<blocks, threads>>>(
        dFrameBuffer, width, height, 4096,
        cam, dTriangles, numTriangles,
        dMaterials, numMaterials,
        dNodes, dObjectIds,
        dRandState);
    cudaDeviceSynchronize();

    std::vector<Vec3> fb(width * height);
    cudaMemcpy(fb.data(), dFrameBuffer, fbMemSize, cudaMemcpyDeviceToHost);

    std::cout << "[Info] Processing and saving image..." << std::endl;
    std::vector<unsigned char> image(width * height * 3);

    for (int j = 0; j < height; ++j)
    {
        for (int i = 0; i < width; ++i)
        {
            size_t pixel_index = (height - 1 - j) * width + i;
            Vec3 pixel = fb[pixel_index];

            float r = sqrt(pixel.x);
            float g = sqrt(pixel.y);
            float b = sqrt(pixel.z);

            // Gamma Correction
            int ir = int(255.99f * (r > 1.0f ? 1.0f : (r < 0.0f ? 0.0f : r)));
            int ig = int(255.99f * (g > 1.0f ? 1.0f : (g < 0.0f ? 0.0f : g)));
            int ib = int(255.99f * (b > 1.0f ? 1.0f : (b < 0.0f ? 0.0f : b)));

            size_t img_index = (j * width + i) * 3;
            image[img_index + 0] = static_cast<unsigned char>(ir);
            image[img_index + 1] = static_cast<unsigned char>(ig);
            image[img_index + 2] = static_cast<unsigned char>(ib);
        }
    }

    if (stbi_write_png("tinypt-lbvh.png", width, height, 3, image.data(), width * 3))
    {
        std::cout << "[Info] Image saved successfully: tinypt.png" << std::endl;
    }
    else
    {
        std::cerr << "[Error] Failed to save image!" << std::endl;
    }

    cudaFree(dTriangles);
    cudaFree(dMaterials);
    cudaFree(dFrameBuffer);
    cudaFree(dRandState);
    return 0;
}