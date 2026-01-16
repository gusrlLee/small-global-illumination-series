#ifndef __TINY_PATH_TRACING_MATERIAL_HEADER__
#define __TINY_PATH_TRACING_MATERIAL_HEADER__

#include "pch.cuh"
#include "vec3.cuh"

enum Type {
    eLAMBERTIAN,
    eDIELECTRIC,
    eSPECULAR,
    eDIFFUSE_LIGHT,
};

struct Material 
{
    Type type;
    Vec3 albedo;
    Vec3 emission;

    __host__ __device__ Material() : type(eLAMBERTIAN), albedo(Vec3(0,0,0)), emission(Vec3(0,0,0)) {}
};

#endif