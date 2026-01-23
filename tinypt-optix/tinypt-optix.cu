#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#include "structs.cuh"

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

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct TriangleMesh
{
    std::vector<float3> vertices;
    std::vector<uint3> indices;

    std::vector<unsigned int> matIndices;
    std::vector<MaterialData> materials;
};

bool loadModel(const std::string &filename, const std::string &mtlDir, TriangleMesh &mesh)
{
    tinyobj::ObjReaderConfig cfg;
    cfg.mtl_search_path = mtlDir;
    cfg.triangulate = true;

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(filename, cfg))
    {
        if (!reader.Error().empty())
        {
            std::cerr << "TinyObjReader Error: " << reader.Error() << std::endl;
        }
        return false;
    }

    if (!reader.Warning().empty())
    {
        std::cerr << "TinyObjReader Warning: " << reader.Warning() << std::endl;
    }

    auto &attrib = reader.GetAttrib();
    auto &shapes = reader.GetShapes();
    auto &materials = reader.GetMaterials();

    for (const auto& mat : materials) 
    {
        MaterialData m;
        m.diffuse = make_float3(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
        m.emission = make_float3(mat.emission[0], mat.emission[1], mat.emission[2]);
        mesh.materials.push_back(m);
    }

    if (mesh.materials.empty()) 
    {
        MaterialData m = { make_float3(0.7f, 0.7f, 0.7f), make_float3(0.0f, 0.0f, 0.0f) };
        mesh.materials.push_back(m);
    }

    for (const auto &shape : shapes)
    {
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
        {
            int fv = 3;
            // vertex
            for (size_t v = 0; v < fv; v++)
            {
                tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
                tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
                tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
                mesh.vertices.push_back(make_float3(vx, vy, vz));
            }

            // index
            unsigned int current_v_count = (unsigned int)mesh.vertices.size();
            mesh.indices.push_back(make_uint3(
                current_v_count - 3,
                current_v_count - 2,
                current_v_count - 1));

            int matId = shape.mesh.material_ids[f];
            if (matId < 0) matId = 0; // 재질 없으면 0번 사용
            mesh.matIndices.push_back((unsigned int)matId);

            index_offset += fv;
        }
    }

    std::cout << "Model Loaded: " << filename << "\n"
              << "  Vertices: " << mesh.vertices.size() << "\n"
              << "  Triangles: " << mesh.indices.size() << "\n" 
              << "  Materials: " << mesh.materials.size() << "\n"
              << "  MatIndices: " << mesh.matIndices.size() << std::endl;

    return true;
}

OptixTraversableHandle buildAccelerationStructure(
    OptixDeviceContext context,
    const TriangleMesh &mesh,
    CUdeviceptr &gasOutputBuffer,
    CUdeviceptr &vertices,
    CUdeviceptr &indices)
{
    const size_t verticeSize = sizeof(float3) * mesh.vertices.size();
    const size_t indexSize = sizeof(uint3) * mesh.indices.size();

    CUDA_CHECK(cudaMalloc((void **)&vertices, verticeSize));
    CUDA_CHECK(cudaMemcpy((void*)vertices, mesh.vertices.data(), verticeSize, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)&indices, indexSize));
    CUDA_CHECK(cudaMemcpy((void*)indices, mesh.indices.data(), indexSize, cudaMemcpyHostToDevice));

    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    buildInput.triangleArray.vertexBuffers = &vertices; 
    buildInput.triangleArray.numVertices   = static_cast<unsigned int>(mesh.vertices.size());
    buildInput.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = sizeof(float3);

    buildInput.triangleArray.indexBuffer   = indices;
    buildInput.triangleArray.numIndexTriplets = static_cast<unsigned int>(mesh.indices.size());
    buildInput.triangleArray.indexFormat   = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.indexStrideInBytes = sizeof(uint3);

    uint32_t triangleInputFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    buildInput.triangleArray.flags = triangleInputFlags;
    buildInput.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context, 
        &accelOptions, 
        &buildInput, 
        1, // num build inputs
        &gasBufferSizes
    ));

    CUdeviceptr tempBuf = 0;
    CUDA_CHECK(cudaMalloc((void**)&tempBuf, gasBufferSizes.tempSizeInBytes));

    CUDA_CHECK(cudaMalloc((void**)&gasOutputBuffer, gasBufferSizes.outputSizeInBytes));

    // 5. 실제 빌드 실행 (GPU에서 BVH 생성)
    OptixTraversableHandle handle;
    OPTIX_CHECK(optixAccelBuild(
        context,
        0, // stream
        &accelOptions,
        &buildInput,
        1, // num build inputs
        tempBuf,
        gasBufferSizes.tempSizeInBytes,
        gasOutputBuffer,
        gasBufferSizes.outputSizeInBytes,
        &handle,
        nullptr, // emitted property list (compaction 나중에)
        0        // num emitted properties
    ));

    // 임시 버퍼는 빌드 끝나면 바로 해제
    CUDA_CHECK(cudaFree((void*)tempBuf));

    return handle;
}

struct EmptyData
{
};

typedef SbtRecord<EmptyData> RayGenSbtRecord;
typedef SbtRecord<EmptyData> MissSbtRecord;
typedef SbtRecord<EmptyData> HitGroupSbtRecord;

std::string readSourceFile(std::string const &filename)
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
        pipelineOpts.pipelineLaunchParamsVariableName = "params";

#ifdef _DEBUG
        pipelineOpts.exceptionFlags = OPTIX_EXCEPTION_FLAG_USER | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
        // pipelineOpts.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#else
        pipelineOpts.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
        // pipelineOpts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        pipelineOpts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;

        OptixModuleCompileOptions moduleOpts = {};
#ifdef _DEBUG
        moduleOpts.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        moduleOpts.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        moduleOpts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
        moduleOpts.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
        moduleOpts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif

        OptixModule module = nullptr;
        std::string ptx_code = readSourceFile("./build/tinypt-optix/ptx/device_programs.ptx");

        char log[2048];
        size_t sizeof_log = sizeof(log);

        OPTIX_CHECK(optixModuleCreate(
            context,
            &moduleOpts,
            &pipelineOpts,
            ptx_code.c_str(),
            ptx_code.size(),
            log,
            &sizeof_log,
            &module));

        // 혹시 모듈 생성 중 경고가 있었다면 출력
        if (sizeof_log > 1)
            std::cout << "Module Create Log: " << log << std::endl;
        std::cout << "OptiX Module created successfully!" << std::endl;

        // Program Group create
        OptixProgramGroupOptions programGroupOpts = {};
        std::vector<OptixProgramGroup> programGroups;

        OptixProgramGroup raygenProgGroup = nullptr;
        OptixProgramGroup missProgGroup = nullptr;
        OptixProgramGroup hitGroupProgGroup = nullptr;

        OptixProgramGroupDesc raygenDesc = {};
        raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygenDesc.raygen.module = module;
        raygenDesc.raygen.entryFunctionName = "__raygen__rg";

        OptixProgramGroupDesc missDesc = {};
        missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        missDesc.miss.module = module;
        missDesc.miss.entryFunctionName = "__miss__ms";

        OptixProgramGroupDesc hitGroupDesc = {};
        hitGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitGroupDesc.hitgroup.moduleCH = module;
        hitGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";

        sizeof_log = sizeof(log);

        // ray generation group
        OPTIX_CHECK(optixProgramGroupCreate(
            context, &raygenDesc, 1, &programGroupOpts,
            log, &sizeof_log, &raygenProgGroup));
        if (sizeof_log > 1)
            std::cout << "RayGen Group Log: " << log << std::endl;
        programGroups.push_back(raygenProgGroup);

        // miss group
        OPTIX_CHECK(optixProgramGroupCreate(
            context, &missDesc, 1, &programGroupOpts,
            log, &sizeof_log, &missProgGroup));
        if (sizeof_log > 1)
            std::cout << "Miss Group Log: " << log << std::endl;
        programGroups.push_back(missProgGroup);

        // hit group
        OPTIX_CHECK(optixProgramGroupCreate(
            context, &hitGroupDesc, 1, &programGroupOpts,
            log, &sizeof_log, &hitGroupProgGroup));
        if (sizeof_log > 1)
            std::cout << "HitGroup Group Log: " << log << std::endl;
        programGroups.push_back(hitGroupProgGroup);

        std::cout << "Program Groups created successfully!" << std::endl;

        // pipeline
        OptixPipelineLinkOptions pipelineLinkOpts = {};
        pipelineLinkOpts.maxTraceDepth = 2;

        OptixPipeline pipeline = nullptr;
        sizeof_log = sizeof(log);

        OPTIX_CHECK(optixPipelineCreate(
            context,
            &pipelineOpts,
            &pipelineLinkOpts,
            programGroups.data(),
            (unsigned int)programGroups.size(),
            log,
            &sizeof_log,
            &pipeline));

        if (sizeof_log > 1)
            std::cout << "Pipeline Create Log: " << log << std::endl;
        std::cout << "OptiX Pipeline created successfully!" << std::endl;

        // create shader binding table
        RayGenSbtRecord rgSbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[0], &rgSbt));

        CUdeviceptr raygenRecordPtr;
        size_t raygenRecordSize = sizeof(RayGenSbtRecord);
        CUDA_CHECK(cudaMalloc((void **)&raygenRecordPtr, raygenRecordSize));
        CUDA_CHECK(cudaMemcpy((void *)raygenRecordPtr, &rgSbt, raygenRecordSize, cudaMemcpyHostToDevice));

        MissSbtRecord msSbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[1], &msSbt));

        CUdeviceptr missRecordPtr;
        size_t missRecordSize = sizeof(MissSbtRecord);
        CUDA_CHECK(cudaMalloc((void **)&missRecordPtr, missRecordSize));
        CUDA_CHECK(cudaMemcpy((void *)missRecordPtr, &msSbt, missRecordSize, cudaMemcpyHostToDevice));

        HitGroupSbtRecord hgSbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[2], &hgSbt));

        CUdeviceptr hitGroupRecordPtr;
        size_t hitGroupRecordSize = sizeof(HitGroupSbtRecord);
        CUDA_CHECK(cudaMalloc((void **)&hitGroupRecordPtr, hitGroupRecordSize));
        CUDA_CHECK(cudaMemcpy((void *)hitGroupRecordPtr, &hgSbt, hitGroupRecordSize, cudaMemcpyHostToDevice));

        OptixShaderBindingTable sbt = {};
        sbt.raygenRecord = raygenRecordPtr;

        sbt.missRecordBase = missRecordPtr;
        sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
        sbt.missRecordCount = 1;

        sbt.hitgroupRecordBase = hitGroupRecordPtr;
        sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        sbt.hitgroupRecordCount = 1;

        std::cout << "SBT built successfully!" << std::endl;

        std::string inputPath = "scenes/CornellBox/CornellBox-Original.obj";
        std::filesystem::path objPath(inputPath);

        if (!std::filesystem::exists(objPath))
        {
            std::cerr << "[Error]: File not found - " << inputPath << std::endl;
            return -1;
        }

        std::string objFp = objPath.string();
        std::string mtlDir = objPath.parent_path().string() + "/";

        TriangleMesh mesh;
        if (!loadModel(objFp, mtlDir, mesh)) 
        { 
            std::cerr << "Failed to load model." << std::endl;
            return -1;
        }

        CUdeviceptr d_gas_output_buffer = 0;
        CUdeviceptr d_vertices = 0;
        CUdeviceptr d_indices = 0;

        OptixTraversableHandle handle = buildAccelerationStructure(
            context, 
            mesh, 
            d_gas_output_buffer, 
            d_vertices, 
            d_indices
        );

        std::cout << "Acceleration Structure built!" << std::endl;
        CUdeviceptr d_materials = 0;
        size_t materialSize = sizeof(MaterialData) * mesh.materials.size();
        CUDA_CHECK(cudaMalloc((void**)&d_materials, materialSize));
        CUDA_CHECK(cudaMemcpy((void*)d_materials, mesh.materials.data(), materialSize, cudaMemcpyHostToDevice));

        CUdeviceptr d_mat_indices = 0;
        size_t matIndexSize = sizeof(unsigned int) * mesh.matIndices.size();
        CUDA_CHECK(cudaMalloc((void**)&d_mat_indices, matIndexSize));
        CUDA_CHECK(cudaMemcpy((void*)d_mat_indices, mesh.matIndices.data(), matIndexSize, cudaMemcpyHostToDevice));

        int width = 800;
        int height = 800;

        uchar4 *dImage = nullptr;
        CUDA_CHECK(cudaMalloc((void **)&dImage, width * height * sizeof(uchar4)));

        float4 *dAccum = nullptr;
        CUDA_CHECK(cudaMalloc((void **)&dAccum, width * height * sizeof(float4)));
        CUDA_CHECK(cudaMemset(dAccum, 0, width * height * sizeof(float4)));

        Params params;
        params.image = dImage;
        params.accum_buffer = dAccum;
        params.width = width;
        params.height = height;
        params.handle = handle;
        params.vertices = (float3*)d_vertices;
        params.indices = (uint3*)d_indices;
        params.materials = (MaterialData*)d_materials;
        params.matIndices = (unsigned int*)d_mat_indices;

        CUdeviceptr dParams;
        CUDA_CHECK(cudaMalloc((void **)&dParams, sizeof(Params)));
        CUDA_CHECK(cudaMemcpy((void *)dParams, &params, sizeof(Params), cudaMemcpyHostToDevice));

        std::cout << "Launching OptiX..." << std::endl;
        int samples = 4096;
    // 2. 렌더링 루프 (Progressive Update)
        for (int i = 0; i < samples; ++i) {
            params.frame_index = i; // 프레임 번호 업데이트

            // 변경된 params를 GPU로 복사
            CUDA_CHECK(cudaMemcpy((void *)dParams, &params, sizeof(Params), cudaMemcpyHostToDevice));

            // 커널 실행
            OPTIX_CHECK(optixLaunch(
                pipeline,
                0,
                dParams,
                sizeof(Params),
                &sbt,
                width,
                height,
                1));

            // 진행 상황 표시 (선택 사항)
            if ((i + 1) % 100 == 0) {
                std::cout << "Rendered " << (i + 1) << " / " << samples << " samples" << std::endl;
                cudaDeviceSynchronize(); 
            }
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "OptiX Launch finished!" << std::endl;

        std::vector<uchar4> hImage(width * height); // CPU 쪽 임시 저장소
        CUDA_CHECK(cudaMemcpy(hImage.data(), dImage, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost));
        std::string outfile = "tinypt-optix.png";
        int stride = width * sizeof(uchar4);
        stbi_write_png(outfile.c_str(), width, height, 4, hImage.data(), stride);
        std::cout << "Saved image to " << outfile << std::endl;


        CUDA_CHECK(cudaFree((void*)d_materials));
        CUDA_CHECK(cudaFree((void*)d_mat_indices));

        CUDA_CHECK(cudaFree((void*)d_gas_output_buffer));
        CUDA_CHECK(cudaFree((void*)d_vertices));
        CUDA_CHECK(cudaFree((void*)d_indices));

        CUDA_CHECK(cudaFree((void *)raygenRecordPtr));
        CUDA_CHECK(cudaFree((void *)missRecordPtr));
        CUDA_CHECK(cudaFree((void *)hitGroupRecordPtr));

        OPTIX_CHECK(optixPipelineDestroy(pipeline));

        OPTIX_CHECK(optixProgramGroupDestroy(hitGroupProgGroup));
        OPTIX_CHECK(optixProgramGroupDestroy(missProgGroup));
        OPTIX_CHECK(optixProgramGroupDestroy(raygenProgGroup));

        OPTIX_CHECK(optixModuleDestroy(module));
        OPTIX_CHECK(optixDeviceContextDestroy(context));
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }
    return 0;
}