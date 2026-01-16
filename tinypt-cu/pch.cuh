#ifndef __TINY_PATH_TRACING_PRE_COMPLIE_HEADER__
#define __TINY_PATH_TRACING_PRE_COMPLIE_HEADER__


#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#define PI 3.1415926535897932385
#define INV_PI 0.31830988618379067154


#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobjloader/tiny_obj_loader.h"

#endif // __TINY_PATH_TRACING_PRE_COMPLIE_HEADER__