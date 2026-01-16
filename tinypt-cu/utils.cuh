#ifndef __TINY_PATH_TRACING_UTILS_HEADER__
#define __TINY_PATH_TRACING_UTILS_HEADER__

#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include "tinyobjloader/tiny_obj_loader.h"

#include "scene.cuh"
#include "triangle.cuh"
#include "material.cuh"

namespace utils
{
    Scene loadObj(const std::string &fp, const std::string &mtlFp);
}


#endif // __TINY_PATH_TRACING_UTILS_HEADER__