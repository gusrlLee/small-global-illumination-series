#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#define OPTIX_CHECK(call)                                                                             \
    {                                                                                                 \
        OptixResult res = call;                                                                       \
        if (res != OPTIX_SUCCESS)                                                                     \
        {                                                                                             \
            fprintf(stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__); \
            exit(2);                                                                                  \
        }                                                                                             \
    }

#define CUDA_CHECK(call)                                                                             \
    {                                                                                                \
        cudaError_t res = call;                                                                      \
        if (res != cudaSuccess)                                                                      \
        {                                                                                            \
            fprintf(stderr, "CUDA call (%s) failed with code %d (line %d)\n", #call, res, __LINE__); \
            exit(2);                                                                                 \
        }                                                                                            \
    }

static void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
              << message << "\n";
}

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


std::string readSourceFile(std::string const& filename)
{
    std::ifstream input(filename.c_str());
    if (!input.good())
    {
        std::runtime_error("Couldn't open file : " + filename);
    }

    std::stringstream source;
    source << input.rdbuf();
    return source.str();
}

int main(int argc, char *argv[])
{
    try
    {
        std::cout << "Initalizing tinypt-optix..." << std::endl;
        CUDA_CHECK(cudaFree(0));
        OPTIX_CHECK(optixInit());

        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &context_log_cb;
        options.logCallbackLevel = 4; // 0 : disable, 4 : info (detail)

#ifdef _DEBUG
        options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#else
        options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
#endif
        // context create
        OptixDeviceContext context = nullptr;
        CUcontext cuCtx = 0; // 0 means current CUDA context 
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
        std::cout << "OptiX Context created successfully!" << std::endl;

        // pipeline setting 
        OptixPipelineCompileOptions pipelineOpts = {};
        pipelineOpts.numPayloadValues = 2;
        pipelineOpts.numAttributeValues = 2;

#ifdef _DEBUG
        pipelineOpts.exceptionFlags = OPTIX_EXCEPTION_FLAG_USER | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
        // pipelineOpts.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#else
        pipelineOpts.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
        pipelineOpts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;

        OptixModuleCompileOptions moduleOpts = {};
#ifdef _DEBUG
        moduleOpts.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        moduleOpts.optLevel         = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        moduleOpts.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
        moduleOpts.optLevel         = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
        moduleOpts.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif

        OptixModule module = nullptr;
        std::string ptx_code = readSourceFile( "./build/tinypt-optix/ptx/device_programs.ptx" );

        char log[2048];
        size_t sizeof_log = sizeof( log );

        OPTIX_CHECK( optixModuleCreate(
            context,
            &moduleOpts,
            &pipelineOpts,
            ptx_code.c_str(),
            ptx_code.size(),
            log,
            &sizeof_log,
            &module
        ) );
        
        // 혹시 모듈 생성 중 경고가 있었다면 출력
        if( sizeof_log > 1 ) std::cout << "Module Create Log: " << log << std::endl;
        std::cout << "OptiX Module created successfully!" << std::endl;

        // Program Group create
        OptixProgramGroupOptions programGroupOpts = {};
        std::vector<OptixProgramGroup> programGroups;

        OptixProgramGroup raygenProgGroup = nullptr;
        OptixProgramGroup missProgGroup = nullptr;
        OptixProgramGroup hitGroupProgGroup = nullptr;

        OptixProgramGroupDesc raygenDesc  = {};
        raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygenDesc.raygen.module = module;
        raygenDesc.raygen.entryFunctionName = "__raygen__rg";

        OptixProgramGroupDesc missDesc  = {};
        missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        missDesc.miss.module = module;
        missDesc.miss.entryFunctionName = "__miss__ms";

        OptixProgramGroupDesc hitGroupDesc  = {};
        hitGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitGroupDesc.hitgroup.moduleCH = module;
        hitGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";

        sizeof_log = sizeof(log);

        // ray generation group
        OPTIX_CHECK( optixProgramGroupCreate(
            context, &raygenDesc, 1, &programGroupOpts,
            log, &sizeof_log, &raygenProgGroup
        ) );
        if( sizeof_log > 1 ) std::cout << "RayGen Group Log: " << log << std::endl;
        programGroups.push_back(raygenProgGroup);

        // miss group 
        OPTIX_CHECK( optixProgramGroupCreate(
            context, &missDesc, 1, &programGroupOpts,
            log, &sizeof_log, &missProgGroup
        ) );
        if( sizeof_log > 1 ) std::cout << "Miss Group Log: " << log << std::endl;
        programGroups.push_back(missProgGroup);

        // hit group
        OPTIX_CHECK( optixProgramGroupCreate(
            context, &hitGroupDesc, 1, &programGroupOpts,
            log, &sizeof_log, &hitGroupProgGroup
        ) );
        if( sizeof_log > 1 ) std::cout << "HitGroup Group Log: " << log << std::endl;
        programGroups.push_back(hitGroupProgGroup);

        std::cout << "Program Groups created successfully!" << std::endl;

        // pipeline 
        OptixPipelineLinkOptions pipelineLinkOpts = {};
        pipelineLinkOpts.maxTraceDepth = 2;

        OptixPipeline pipeline = nullptr;
        sizeof_log = sizeof(log);

        OPTIX_CHECK( optixPipelineCreate(
            context,
            &pipelineOpts,           
            &pipelineLinkOpts,  
            programGroups.data(),  
            (unsigned int)programGroups.size(), 
            log,
            &sizeof_log,
            &pipeline
        ) );

        if( sizeof_log > 1 ) std::cout << "Pipeline Create Log: " << log << std::endl;
        std::cout << "OptiX Pipeline created successfully!" << std::endl;

        OPTIX_CHECK( optixPipelineDestroy( pipeline ) );
        
        OPTIX_CHECK( optixProgramGroupDestroy( hitGroupProgGroup ) );
        OPTIX_CHECK( optixProgramGroupDestroy( missProgGroup ) );
        OPTIX_CHECK( optixProgramGroupDestroy( raygenProgGroup ) );
        
        OPTIX_CHECK( optixModuleDestroy( module ) );
        OPTIX_CHECK( optixDeviceContextDestroy( context ) );
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }
    return 0;

    // std::string inputPath = (argc > 1) ? argv[1] : "scenes/CornellBox/CornellBox-Original.obj";
    // std::filesystem::path objPath(inputPath);

    // if (!std::filesystem::exists(objPath))
    // {
    //     std::cerr << "[Error]: File not found - " << inputPath << std::endl;
    //     return -1;
    // }

    // std::string objFp = objPath.string();
    // std::string mtlDir = objPath.parent_path().string() + "/";

    // std::cout << "[Info] OBJ Path: " << objFp << std::endl;
    // std::cout << "[Info] MTL Dir : " << mtlDir << std::endl;

    // // scene info
    // int width = 1000;
    // int height = 1000;
    // float aspectRatio = float(width) / float(height);
    // Scene scn = loadScene(objFp, mtlDir);

    // // create camera
    // // lookFrom, lookAt, vup, vfov, aspectRatio
    // Camera cam(Vec3(0.0f, 1.0f, 4.0f), Vec3(0.0f, 1.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), 40.0f, aspectRatio);

    // // calculation of memory for GPU
    // int numTriangles = scn.triangles.size();
    // int numMaterials = scn.materials.size();
    // size_t triangleMemSize = numTriangles * sizeof(Triangle);
    // size_t materialMemSize = numMaterials * sizeof(Material);
    // size_t fbMemSize = width * height * sizeof(Vec3);
    // size_t randStateSize = width * height * sizeof(curandState);

    // Triangle *dTriangles;
    // Material *dMaterials;
    // Vec3 *dFrameBuffer;
    // curandState *dRandState;

    // cudaMalloc((void **)&dTriangles, triangleMemSize);
    // cudaMalloc((void **)&dMaterials, materialMemSize);
    // cudaMalloc((void **)&dFrameBuffer, fbMemSize);
    // cudaMalloc((void **)&dRandState, randStateSize);

    // cudaMemcpy(dTriangles, scn.triangles.data(), triangleMemSize, cudaMemcpyHostToDevice);
    // cudaMemcpy(dMaterials, scn.materials.data(), materialMemSize, cudaMemcpyHostToDevice);

    // int tx = 8, ty = 8;
    // dim3 blocks(width / tx + 1, height / ty + 1);
    // dim3 threads(tx, ty);

    // initRandState<<<blocks, threads>>>(width, height, dRandState);
    // cudaDeviceSynchronize();

    // render<<<blocks, threads>>>(
    //     dFrameBuffer, width, height, 20000,
    //     cam, dTriangles, numTriangles,
    //     dMaterials, numMaterials,
    //     dRandState);
    // cudaDeviceSynchronize();

    // std::vector<Vec3> fb(width * height);
    // cudaMemcpy(fb.data(), dFrameBuffer, fbMemSize, cudaMemcpyDeviceToHost);

    // std::cout << "[Info] Processing and saving image..." << std::endl;
    // std::vector<unsigned char> image(width * height * 3);

    // for (int j = 0; j < height; ++j)
    // {
    //     for (int i = 0; i < width; ++i)
    //     {
    //         size_t pixel_index = (height - 1 - j) * width + i;
    //         Vec3 pixel = fb[pixel_index];

    //         float r = sqrt(pixel.x);
    //         float g = sqrt(pixel.y);
    //         float b = sqrt(pixel.z);

    //         // Gamma Correction
    //         int ir = int(255.99f * (r > 1.0f ? 1.0f : (r < 0.0f ? 0.0f : r)));
    //         int ig = int(255.99f * (g > 1.0f ? 1.0f : (g < 0.0f ? 0.0f : g)));
    //         int ib = int(255.99f * (b > 1.0f ? 1.0f : (b < 0.0f ? 0.0f : b)));

    //         size_t img_index = (j * width + i) * 3;
    //         image[img_index + 0] = static_cast<unsigned char>(ir);
    //         image[img_index + 1] = static_cast<unsigned char>(ig);
    //         image[img_index + 2] = static_cast<unsigned char>(ib);
    //     }
    // }

    // if (stbi_write_png("tinypt.png", width, height, 3, image.data(), width * 3))
    // {
    //     std::cout << "[Info] Image saved successfully: tinypt.png" << std::endl;
    // }
    // else
    // {
    //     std::cerr << "[Error] Failed to save image!" << std::endl;
    // }

    // cudaFree(dTriangles);
    // cudaFree(dMaterials);
    // cudaFree(dFrameBuffer);
    // cudaFree(dRandState);
    // return 0;
}