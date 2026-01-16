#include "utils.cuh"


#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobjloader/tiny_obj_loader.h"

namespace utils
{
    Scene loadObj(const std::string &fp, const std::string &mtlFp)
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
                        attrib.vertices[3 * idx.vertex_index + 2]
                    );
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
}
