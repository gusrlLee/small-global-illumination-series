#include <cuda_runtime.h>
#include <iostream>
#include <filesystem>
#include "utils.cuh"

int main(int argc, char* argv[]) 
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

    Scene scn = utils::loadObj(objFp, mtlDir);

    std::cout << "[Info] Scene data size = " << scn.materials.size() << std::endl;
    std::cout << "[Info] Triangle data size = " << scn.triangles.size() << std::endl;

    for (int i = 0; i < scn.materials.size(); i++)
    {
        std::cout << "[Info] albedo = " << scn.materials[i].albedo.x << ", " << scn.materials[i].albedo.y << ", " << scn.materials[i].albedo.z << std::endl;
        std::cout << "[Info] emission = " << scn.materials[i].emission.x << ", " << scn.materials[i].emission.y << ", " << scn.materials[i].emission.z << std::endl;
        std::cout << "[Info] type = " << scn.materials[i].type << std::endl;
    }

    return 0;
}