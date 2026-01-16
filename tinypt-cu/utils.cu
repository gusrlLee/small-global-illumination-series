#include "utils.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobjloader/tiny_obj_loader.h"



namespace utils
{
    std::vector<Triangle> loadObj(const std::string &fp, const std::string &mtlFp)
    {
        std::vector<Triangle> triangles;
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
            return triangles; 
        }

        if (!reader.Warning().empty()) 
        {
            std::cerr << "TinyObjReader Warning: " << reader.Warning() << std::endl;
        }

        auto& attrib = reader.GetAttrib();
        auto& shapes = reader.GetShapes();
        auto& materials = reader.GetMaterials();

        for (const auto& shape : shapes)
        {
            size_t idxOffset = 0;
            for (size_t face = 0; face < shape.mesh.num_face_vertices.size(); face++)
            {
                
            }
        }
    }

}
