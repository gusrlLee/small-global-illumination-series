#include "Vector3.cuh"
#include <vector>

#define M_PI 3.14159265359f

__host__ __device__ inline float Clamp(float x) { return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f
                                                                                    : x; }
__host__ __device__ inline int ToInt(float x){ return int(pow(Clamp(x), 1 / 2.2) * 255 + .5); }

__device__ unsigned int WangHash(unsigned int seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

__device__ static float GetRandom(unsigned int *seed0, unsigned int *seed1)
{
    *seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16); // hash the seeds using bitwise AND and bitshifts
    *seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);
    unsigned int ires = ((*seed0) << 16) + (*seed1);

    // Convert to float
    union
    {
        float f;
        unsigned int ui;
    } res;

    res.ui = (ires & 0x007fffff) | 0x40000000; // bitwise AND, bitwise OR
    return (res.f - 2.f) / 2.f;
}

class Camera
{
public:
    __device__ Camera(Vector3 position, Vector3 direction) : m_Position(position), m_Direction(direction) {}
    Vector3 m_Position;
    Vector3 m_Direction;
    float m_Fov;
};

class Ray
{
public:
    __device__ Ray(Vector3 origin, Vector3 direction) : m_Origin(origin), m_Direction(direction) {}
    __device__ Vector3 At(float t) const { return m_Origin + t * m_Direction; }

    Vector3 m_Origin;
    Vector3 m_Direction;
};

enum Refl_t
{
    DIFF,
    SPEC,
    REFR
};

struct Sphere
{
    float m_Radius;
    Vector3 m_Position, m_Emissive, m_Color;
    Refl_t m_MaterialType;

    __device__ bool IsLight() const { return (m_Emissive.x() > 0.0f) | (m_Emissive.y() > 0.0f) | (m_Emissive.z() > 0.0f); }

    __device__ float Intersect(const Ray &r) const
    {
        Vector3 op = m_Position - r.m_Origin;                   // distance from ray.orig to center sphere
        float t, epsilon = 0.0001f;                             // epsilon required to prevent floating point precision artefacts
        float b = Dot(op, r.m_Direction);                       // b in quadratic equation
        float disc = b * b - Dot(op, op) + m_Radius * m_Radius; // discriminant quadratic equation
        if (disc < 0)
            return 0; // if disc < 0, no real solution (we're not interested in complex roots)
        else
            disc = sqrtf(disc);                                                   // if disc >= 0, check for solutions using negative and positive discriminant
        return (t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0); // pick closest point in front of ray origin
    }

    __device__ void OnSample(Vector3 &sample, float &pdf, unsigned int *s1, unsigned int *s2) const {
        float z = 1.0f - 2.0f * GetRandom(s1, s2);
        float r = sqrtf(max(0.0f, 1.0f - z * z));
        float phi = 2.0f * M_PI * GetRandom(s1, s2);

        Vector3 lPoint = Vector3(r * cosf(phi), r * sinf(phi), z); // sample point on unit sphere
        sample = m_Position + lPoint * m_Radius;

        float area = 4.0f * M_PI * m_Radius * m_Radius;
        pdf = 1.0f / area;
    }

    __device__ float PdfLight(const Vector3& hitPoint, const Vector3 rayDir) const 
    {
        if (!IsLight()) return 0.0f;
        Vector3 op = m_Position - hitPoint;
        float distSq = op.SquareLength();
        float dist = sqrtf(distSq);

        float cosTheta = 1.0f;
        float t = Intersect(Ray(hitPoint, rayDir));
        if (t <= 0.0f) return 0.0f;

        Vector3 lightPos = hitPoint + t * rayDir;
        Vector3 lightNormal = Normalize(lightPos - m_Position);
        float cosLight = fabs(Dot(lightNormal, -rayDir));

        float area = 4.0f * M_PI * m_Radius * m_Radius;
        float pdfArea = 1.0f / area;

        return pdfArea * (t * t) / cosLight;
    }
};

__constant__ Sphere spheres[9];
__constant__ Sphere lights[1];

__device__ inline float PowerHeuristic(float fPdf, float gPdf)
{
    float f2 = fPdf * fPdf;
    float g2 = gPdf * gPdf;
    return f2 / (f2 + g2 + 1e-5f);
}

__device__ inline bool TraceRay(const Ray &r, float &t, int &id)
{
    float n = sizeof(spheres) / sizeof(Sphere), d, inf = t = 1e20;
    for (int i = int(n); i--;)
    {
        if ((d = spheres[i].Intersect(r)) && d < t)
        {
            t = d;
            id = i;
        }
    }
    return t < inf;
}

__device__ inline bool TraceShadowRay(const Ray &r, float &tMax)
{
    float n = sizeof(spheres) / sizeof(Sphere), d;

    float tChk = tMax - 0.0001f;
    for (int i = int(n); i--;)
    {
        if ((d = spheres[i].Intersect(r)) && d > 0.001f && d < tChk) 
            return true; // this shadow ray
    }

    return false; // this is not shadow ray
}

__device__ Vector3 Radiance(Ray &r, unsigned int *s1, unsigned int *s2, int maxDepth)
{
    Vector3 color = Vector3(0.0f);
    Vector3 thp = Vector3(1.0f);

    float lastBsdfPDf = 0.0f;
    bool lastSpecular = true;

    for (int d = 0; d < maxDepth; d++)
    {
        float t;
        int id = -1;

        if (!TraceRay(r, t, id))
        {
            break;
        }
        
        const Sphere &obj = spheres[id];

        // if obj is light, we need to stop traversal step.
        if (obj.IsLight()) 
        {
            if (lastSpecular)
            {
                color += thp * obj.m_Emissive;
            }
            else
            {
                float distSq = t * t;
                Vector3 lightNormal = Normalize(r.At(t) - obj.m_Position);
                float lightAngleCosine = fabs(Dot(lightNormal, -r.m_Direction));

                float lightArea = 4.0f * M_PI * obj.m_Radius * obj.m_Radius;
                float lightPdfArea = 1.0f / lightArea;
                float lightPdfSolid = lightPdfArea * (distSq / lightAngleCosine);

                float weight = PowerHeuristic(lastBsdfPDf, lightPdfSolid);
                color += thp * obj.m_Emissive * weight;
            }

            break; // terminate path tracing
        }

        Vector3 x = r.At(t);
        Vector3 n = Normalize(x - obj.m_Position);
        Vector3 nl = Dot(n, r.m_Direction) < 0 ? n : -n;

        {
            const Sphere &light = lights[0];

            Vector3 lightPos;
            float lightPdfArea;
            light.OnSample(lightPos, lightPdfArea, s1, s2);
            
            Vector3 toLight = lightPos - x;
            float distSq = toLight.SquareLength();
            float dist = sqrtf(distSq);
            Vector3 lightDir = Normalize(toLight);
            float cosTheta = Dot(nl, lightDir);

            if (cosTheta > 0.0f)
            {
                Vector3 lightNormal = Normalize(lightPos - light.m_Position);
                float lightCosine = Dot(-lightDir, lightNormal);

                if (lightCosine > 0.0f)
                {
                    Ray shadowRay(x + nl * 0.001f, lightDir);
                    float shadowDist = dist - 0.002f;
                    if (!TraceShadowRay(shadowRay, shadowDist))
                    {
                        float lightPdfSolid = lightPdfArea * (distSq / lightCosine);
                        float bsdfPdf = cosTheta / M_PI;
                        float weight = PowerHeuristic(lightPdfSolid, bsdfPdf);
                        Vector3 brdf = obj.m_Color * (1.0f / M_PI);

                        color += thp * light.m_Emissive * brdf * cosTheta * weight / lightPdfSolid;
                    }

                }
            }
        }

        float r1 = 2 * M_PI * GetRandom(s1, s2);
        float r2 = GetRandom(s1, s2);
        float r2s = sqrtf(r2);
        
        Vector3 w = nl;
        Vector3 u = Normalize(Cross((fabs(w.x()) > .1 ? Vector3(0, 1, 0) : Vector3(1, 0, 0)), w));
        Vector3 v = Cross(w, u);

        Vector3 nextDir = Normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrtf(1 - r2));

        float cosNext = Dot(nextDir, nl);
        lastBsdfPDf = cosNext / M_PI;
        lastSpecular = false;
        
        r.m_Origin = x + nl * 0.001f;
        r.m_Direction = nextDir;

        thp *= obj.m_Color;
        thp *= Dot(r.m_Direction, nl);
        thp *= 2;
    }

    return color;
}

__global__ void RenderKernel(Vector3 *output, int w, int h, int spp, int maxDepth)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= w) || (y >= h))
        return;

    unsigned int s1 = x * 374761393U + y * 668265263U;
    unsigned int s2 = x * 668265263U + y * 374761393U;

    GetRandom(&s1, &s2);
    GetRandom(&s1, &s2);

    Camera cam(Vector3(50, 52, 295.6), Normalize(Vector3(0, -0.042612, -1)));
    Vector3 cx = Vector3(w * .5135 / h, 0.0f, 0.0f);
    Vector3 cy = Normalize(Cross(cx, cam.m_Direction)) * .5135f;
    Vector3 radiance = Vector3(0.0f);


    for (int s = 0; s < spp; s++)
    {
        Vector3 dir = cam.m_Direction + cx * ((GetRandom(&s1, &s2) + x) / w - .5) + cy * ((GetRandom(&s1, &s2) + y) / h - .5);
        Ray ray = Ray(cam.m_Position + dir * 40, Normalize(dir));
        radiance = radiance + Radiance(ray, &s1, &s2, maxDepth);
    }

    radiance = radiance * (1.0f / spp);

    int pIdx = (h - y - 1) * w + x;
    output[pIdx] = Vector3(Clamp(radiance.r()), Clamp(radiance.g()), Clamp(radiance.b()));
}

int main()
{
    Sphere spheres_h[9] = {
        {1e5f, {1e5f + 1.0f, 40.8f, 81.6f}, {0.0f, 0.0f, 0.0f}, {0.75f, 0.25f, 0.25f}, DIFF},    // Left
        {1e5f, {-1e5f + 99.0f, 40.8f, 81.6f}, {0.0f, 0.0f, 0.0f}, {.25f, .25f, .75f}, DIFF},     // Rght
        {1e5f, {50.0f, 40.8f, 1e5f}, {0.0f, 0.0f, 0.0f}, {.75f, .75f, .75f}, DIFF},              // Back
        {1e5f, {50.0f, 40.8f, -1e5f + 600.0f}, {0.0f, 0.0f, 0.0f}, {1.00f, 1.00f, 1.00f}, DIFF}, // Frnt
        {1e5f, {50.0f, 1e5f, 81.6f}, {0.0f, 0.0f, 0.0f}, {.75f, .75f, .75f}, DIFF},              // Botm
        {1e5f, {50.0f, -1e5f + 81.6f, 81.6f}, {0.0f, 0.0f, 0.0f}, {.75f, .75f, .75f}, DIFF},     // Top
        {16.5f, {27.0f, 16.5f, 47.0f}, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, DIFF},            // small sphere 1
        {16.5f, {73.0f, 16.5f, 78.0f}, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, DIFF},            // small sphere 2
        {600.0f, {50.0f, 681.6f - .77f, 81.6f}, {2.0f, 1.8f, 1.6f}, {0.0f, 0.0f, 0.0f}, DIFF}    // Light
        // {1.5f, {50.0f, 80.0f, 81.6f}, {120.0f, 120.0f, 120.0f}, {0.0f, 0.0f, 0.0f}, DIFF}
    };

    Sphere lights_h[1] = {
        {600.0f, {50.0f, 681.6f - .77f, 81.6f}, {2.0f, 1.8f, 1.6f}, {0.0f, 0.0f, 0.0f}, DIFF}    // Light
        // {1.5f, {50.0f, 80.0f, 81.6f}, {120.0f, 120.0f, 120.0f}, {0.0f, 0.0f, 0.0f}, DIFF}
    };

    cudaError_t err = cudaMemcpyToSymbol(spheres, spheres_h, sizeof(Sphere) * 9);
    if (err != cudaSuccess) 
    {
        printf("Constant memory copy failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMemcpyToSymbol(lights, lights_h, sizeof(Sphere) * 1);
    if (err != cudaSuccess) 
    {
        printf("Constant memory copy failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    int w = 512;
    int h = 512;
    int spp = 4;
    int maxDepth = 5;

    Vector3 *output_h = new Vector3[w * h];
    Vector3 *output_d;

    cudaMalloc(&output_d, w * h * sizeof(Vector3));

    dim3 block(8, 8, 1);
    dim3 grid(w / block.x, h / block.y, 1);

    printf("CUDA initialised.\nStart rendering...\n");

    RenderKernel<<<grid, block>>>(output_d, w, h, spp, maxDepth);
    cudaDeviceSynchronize();

    cudaMemcpy(output_h, output_d, w * h * sizeof(Vector3), cudaMemcpyDeviceToHost);
    cudaFree(output_d);

    FILE *f = fopen("smallmis.ppm", "w");
    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    for (int i = 0; i < w * h; i++)
    {
        fprintf(f, "%d %d %d ",
                ToInt(output_h[i].r()),
                ToInt(output_h[i].g()),
                ToInt(output_h[i].b()));
    }
    fclose(f);
    printf("Done.\n");

    delete[] output_h;
    return 0;
}