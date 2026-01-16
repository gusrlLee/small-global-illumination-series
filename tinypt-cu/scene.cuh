#ifndef __TINY_PATH_TRACING_SCENE_HEADER__
#define __TINY_PATH_TRACING_SCENE_HEADER__

#include "triangle.cuh"
#include "material.cuh"

struct Scene 
{
    std::vector<Triangle> triangles;
    std::vector<Material> materials;
};

#endif