#pragma once 

#include <prayground/optix/cuda/device_util.cuh>

#include <prayground/core/bsdf.h>
#include <prayground/core/onb.h>
#include <prayground/core/color.h>
#include <prayground/core/interaction.h>
#include <prayground/core/ray.h>

#include <prayground/shape/trianglemesh.h>
#include <prayground/shape/plane.h>
#include <prayground/shape/sphere.h>
#include <prayground/shape/box.h>

#include <prayground/texture/constant.h>

#include <prayground/material/diffuse.h>
#include <prayground/material/dielectric.h>

#include <prayground/emitter/area.h>
#include <prayground/emitter/envmap.h>

#include "params.h"

// To fill the inside of sphere with constant medium
#include "box_medium.h"
#include <prayground/material/isotropic.h>

using namespace prayground;

extern "C" {
__constant__ LaunchParams params;
}

INLINE DEVICE SurfaceInteraction* getSurfaceInteraction()
{
    const uint32_t u0 = getPayload<0>();
    const uint32_t u1 = getPayload<1>();
    return reinterpret_cast<SurfaceInteraction*>(unpackPointer(u0, u1));
}

INLINE DEVICE void trace(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax,
    unsigned int           ray_type,
    SurfaceInteraction*    si
) 
{
    unsigned int u0, u1;
    packPointer( si, u0, u1 );
    optixTrace(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.0f,                // rayTime
        OptixVisibilityMask( 1 ),
        OPTIX_RAY_FLAG_NONE,
        ray_type,        
        1,           
        ray_type,        
        u0, u1 );	
}

// Raygen shader -----------------------------------------------------------------------
static __forceinline__ __device__ void getCameraRay(const CameraData& camera, const float x, const float y, float3& ro, float3& rd)
{
    rd = normalize(x * camera.U + y * camera.V + camera.W);
    ro = camera.origin;
}

extern "C" __device__ void __raygen__pinhole()
{
    const RaygenData* raygen = reinterpret_cast<RaygenData*>(optixGetSbtDataPointer());

    const int frame = params.frame;
    const uint3 idx = optixGetLaunchIndex();
    unsigned int seed = tea<4>(idx.x * params.width + idx.y, frame);

    float3 result = make_float3(0.0f);
    float3 normal = make_float3(0);

    int i = params.samples_per_launch;

    do
    {
        const float2 subpixel_jitter = make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f);

        const float2 d = 2.0f * make_float2(
            (static_cast<float>(idx.x) + subpixel_jitter.x) / static_cast<float>(params.width),
            (static_cast<float>(idx.y) + subpixel_jitter.y) / static_cast<float>(params.height)
        ) - 1.0f;

        float3 ro, rd;
        getCameraRay(raygen->camera, d.x, d.y, ro, rd);

        float3 throughput = make_float3(1.0f);

        SurfaceInteraction si;
        si.seed = seed;
        si.emission = make_float3(0.0f);
        si.albedo = make_float3(0.0f);
        si.trace_terminate = false;
        si.radiance_evaled = false;

        int depth = 0;
        for ( ;; ) {

            if ( depth >= params.max_depth )
				break;

            trace(params.handle, ro, rd, 0.01f, 1e16f, 0, &si);

            if (si.trace_terminate) {
                result += si.emission * throughput;
                break;
            }

            if (depth == 0)
                normal = make_float3(fabs(si.n.x), fabs(si.n.y), fabs(si.n.z));

            optixDirectCall<void, SurfaceInteraction*, void*>(
                si.surface_info.sample_id, 
                &si, 
                si.surface_info.data
            );

            if (si.trace_terminate) {
                result += si.emission * throughput;
                break;
            }
            else 
            {
                throughput *= si.albedo;
            }
            
            ro = si.p;
            rd = si.wo;

            ++depth;
        }
    } while (--i);

    const uint3 launch_index = optixGetLaunchIndex();
    const unsigned int image_index = launch_index.y * params.width + launch_index.x;

    if (result.x != result.x) result.x = 0.0f;
    if (result.y != result.y) result.y = 0.0f;
    if (result.z != result.z) result.z = 0.0f;

    float3 accum_color = result / static_cast<float>(params.samples_per_launch);

    if (frame > 0)
    {
        const float a = 1.0f / static_cast<float>(frame + 1);
        const float3 accum_color_prev = make_float3(params.accum_buffer[image_index]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }
    params.accum_buffer[image_index] = make_float4(accum_color, 1.0f);
    uchar3 color = make_color(accum_color);
    //uchar3 color = make_color(normal, false);
    params.result_buffer[image_index] = make_uchar4(color.x, color.y, color.z, 255);
}

// Miss shader -----------------------------------------------------------------------
extern "C" __device__ void __miss__envmap()
{
    MissData* data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    EnvironmentEmitterData* env = reinterpret_cast<EnvironmentEmitterData*>(data->env_data);
    SurfaceInteraction* si = getSurfaceInteraction();

    Ray ray = getWorldRay();

    const float a = dot(ray.d, ray.d);
    const float half_b = dot(ray.o, ray.d);
    const float c = dot(ray.o, ray.o) - 1e8f*1e8f;
    const float discriminant = half_b * half_b - a*c;

    float sqrtd = sqrtf(discriminant);
    float t = (-half_b + sqrtd) / a;

    float3 p = normalize(ray.at(t));

    float phi = atan2(p.z, p.x);
    float theta = asin(p.y);
    float u = 1.0f - (phi + math::pi) / (2.0f * math::pi);
    float v = 1.0f - (theta + math::pi / 2.0f) / math::pi;
    si->uv = make_float2(u, v);
    si->trace_terminate = true;
    si->surface_info.type = SurfaceType::None;
    si->emission = optixDirectCall<float3, const float2&, void*>(
        env->tex_program_id, si->uv, env->tex_data
        );
}

// Hitgroup shaders -----------------------------------------------------------------------
extern "C" __device__ void __closesthit__mesh()
{
    HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const MeshData* mesh_data = reinterpret_cast<MeshData*>(data->shape_data);

    Ray ray = getWorldRay();
    
    const int prim_id = optixGetPrimitiveIndex();
    const Face face = mesh_data->faces[prim_id];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    const float2 texcoord0 = mesh_data->texcoords[face.texcoord_id.x];
    const float2 texcoord1 = mesh_data->texcoords[face.texcoord_id.y];
    const float2 texcoord2 = mesh_data->texcoords[face.texcoord_id.z];
    const float2 texcoords = (1-u-v)*texcoord0 + u*texcoord1 + v*texcoord2;

    float3 n0 = mesh_data->normals[face.normal_id.x];
	float3 n1 = mesh_data->normals[face.normal_id.y];
	float3 n2 = mesh_data->normals[face.normal_id.z];

    // Linear interpolation of normal by barycentric coordinates.
    float3 local_n = (1.0f-u-v)*n0 + u*n1 + v*n2;
    float3 world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->n = world_n;
    si->t = ray.tmax;
    si->wi = ray.d;
    si->uv = texcoords;
    si->surface_info = data->surface_info;
}

extern "C" __device__ void __intersection__plane()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const PlaneData* plane_data = reinterpret_cast<PlaneData*>(data->shape_data);

    const float2 min = plane_data->min;
    const float2 max = plane_data->max;

    Ray ray = getLocalRay();

    const float t = -ray.o.y / ray.d.y;

    const float x = ray.o.x + t * ray.d.x;
    const float z = ray.o.z + t * ray.d.z;

    float2 uv = make_float2((x - min.x) / (max.x - min.x), (z - min.y) / (max.y - min.y));

    float3 n = make_float3(0, 1, 0);

    if (min.x < x && x < max.x && min.y < z && z < max.y && ray.tmin < t && t < ray.tmax)
        optixReportIntersection(t, 0, float3_as_ints(n), float2_as_ints(uv));
}

extern "C" __device__ void __closesthit__plane()
{
    HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    
    Ray ray = getWorldRay();

    float3 local_n = getFloat3FromAttribute<0>();
    float3 world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);
    float2 uv = getFloat2FromAttribute<3>();

    SurfaceInteraction* si = getSurfaceInteraction();

    si->p = ray.at(ray.tmax);
    si->n = world_n;
    si->t = ray.tmax;
    si->wi = ray.d;
    si->uv = uv;
    si->surface_info = data->surface_info;
}

// Box -------------------------------------------------------------------------------
static INLINE DEVICE float2 getBoxUV(const float3& p, const float3& min, const float3& max, const int axis)
{
    float2 uv;
    int u_axis = (axis + 1) % 3;
    int v_axis = (axis + 2) % 3;

    // axis‚ªY‚ÌŽž‚Í (u: Z, v: X) -> (u: X, v: Z)‚Ö‡”Ô‚ð•Ï‚¦‚é
    if (axis == 1) swap(u_axis, v_axis);

    uv.x = (getByIndex(p, u_axis) - getByIndex(min, u_axis)) / (getByIndex(max, u_axis) - getByIndex(min, u_axis));
    uv.y = (getByIndex(p, v_axis) - getByIndex(min, v_axis)) / (getByIndex(max, v_axis) - getByIndex(min, v_axis));

    return clamp(uv, 0.0f, 1.0f);
}

static INLINE DEVICE int hitBox(
    const BoxData* box_data, 
    const float3& o, const float3& v, 
    const float tmin, const float tmax, 
    SurfaceInteraction& si)
{
    float3 min = box_data->min;
    float3 max = box_data->max;

    float _tmin = tmin, _tmax = tmax;
    int min_axis = -1, max_axis = -1;

    for (int i = 0; i < 3; i++)
    {
        float t0, t1;
        if (getByIndex(v, i) == 0.0f)
        {
            t0 = fminf(getByIndex(min, i) - getByIndex(o, i), getByIndex(max, i) - getByIndex(o, i));
            t1 = fmaxf(getByIndex(min, i) - getByIndex(o, i), getByIndex(max, i) - getByIndex(o, i));
        }
        else
        {
            t0 = fminf((getByIndex(min, i) - getByIndex(o, i)) / getByIndex(v, i),
                       (getByIndex(max, i) - getByIndex(o, i)) / getByIndex(v, i));
            t1 = fmaxf((getByIndex(min, i) - getByIndex(o, i)) / getByIndex(v, i),
                       (getByIndex(max, i) - getByIndex(o, i)) / getByIndex(v, i));
        }
        min_axis = t0 > _tmin ? i : min_axis;
        max_axis = t1 < _tmax ? i : max_axis;

        _tmin = fmaxf(t0, _tmin);
        _tmax = fminf(t1, _tmax);

        if (_tmax < _tmin)
            return -1;
    }

    float3 center = (min + max) / 2.0f;
    if ((tmin < _tmin && _tmin < tmax) && (-1 < min_axis && min_axis < 3))
    {
        float3 p = o + _tmin * v;
        float3 center_axis = p;
        setByIndex(center_axis, min_axis, getByIndex(center, min_axis));
        float3 normal = normalize(p - center_axis);
        float2 uv = getBoxUV(p, min, max, min_axis);
        si.p = p;
        si.n = normal;
        si.uv = uv;
        si.t = _tmin;
        return min_axis;
    }

    if ((tmin < _tmax && _tmax < tmax) && (-1 < max_axis && max_axis < 3))
    {
        float3 p = o + _tmax * v;
        float3 center_axis = p;
        setByIndex(center_axis, max_axis, getByIndex(center, max_axis));
        float3 normal = normalize(p - center_axis);
        float2 uv = getBoxUV(p, min, max, max_axis);
        si.p = p;
        si.n = normal;
        si.uv = uv;
        si.t = _tmax;
        return max_axis;
    }
    return -1;
}

extern "C" __device__ void __intersection__box()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const int prim_id = optixGetPrimitiveIndex();
    BoxData box_data = reinterpret_cast<BoxData*>(data->shape_data)[prim_id];

    Ray ray = getLocalRay();

    SurfaceInteraction si = {};
    int hit_axis = hitBox(&box_data, ray.o, ray.d, ray.tmin, ray.tmax, si);
    if (hit_axis >= 0)
        optixReportIntersection(si.t, 0, float3_as_ints(si.n), float2_as_ints(si.uv));
}

/// From "Ray Tracing: The Next Week" by Peter Shirley
/// @ref: https://raytracing.github.io/books/RayTracingTheNextWeek.html
extern "C" __device__ void __intersection__box_medium()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const int prim_id = optixGetPrimitiveIndex();
    BoxMediumData box_medium_data = reinterpret_cast<BoxMediumData*>(data->shape_data)[prim_id];
    BoxData box_data;
    box_data.min = box_medium_data.min;
    box_data.max = box_medium_data.max;

    Ray ray = getLocalRay();

    SurfaceInteraction* global_si = getSurfaceInteraction();
    unsigned int seed = global_si->seed;

    SurfaceInteraction si1 = {}, si2 = {};
    if (hitBox(&box_data, ray.o, ray.d, -1e16f, 1e16f, si1) < 0) 
        return;
    if (hitBox(&box_data, ray.o, ray.d, si1.t + math::eps, 1e16f, si2) < 0)
        return;

    if (si1.t < ray.tmin) si1.t = ray.tmin;
    if (si2.t > ray.tmax) si2.t = ray.tmax;

    if (si1.t >= si2.t) 
        return;
    
    if (si1.t < 0.0f)
        si1.t = 0.0f;

    const float neg_inv_density = -1.0f / box_medium_data.density;
    const float ray_length = length(ray.d);
    const float distance_inside_boundary = (si2.t - si1.t) * ray_length;
    const float hit_distance = neg_inv_density * logf(rnd(seed));
    global_si->seed = seed;

    if (hit_distance > distance_inside_boundary)
        return;
    
    const float t = si1.t + hit_distance / ray_length;
    optixReportIntersection(t, 0, float3_as_ints(si1.n), float2_as_ints(si1.uv), 0);
}

extern "C" __device__ void __closesthit__box()
{
    HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());

    Ray ray = getWorldRay();

    float3 local_n = getFloat3FromAttribute<0>();
    float2 uv = getFloat2FromAttribute<3>();

    float3 world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);

    SurfaceInteraction* si = getSurfaceInteraction();

    si->p = ray.at(ray.tmax);
    si->n = world_n;
    si->t = ray.tmax;
    si->wi = ray.d;
    si->uv = uv;
    si->surface_info = data->surface_info;
}


extern "C" __device__ void __closesthit__sphere() {
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const SphereData* sphere_data = reinterpret_cast<SphereData*>(data->shape_data);

    Ray ray = getWorldRay();

    float3 local_n = getFloat3FromAttribute<0>();
    float2 uv = getFloat2FromAttribute<3>();
    float3 world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->n = world_n;
    si->t = ray.tmax;
    si->wi = ray.d;
    si->uv = uv;
    si->surface_info = data->surface_info;
}

// Textures -----------------------------------------------------------------------
extern "C" __device__ float3 __direct_callable__constant(const float2& uv, void* tex_data) {
    const ConstantTextureData* constant = reinterpret_cast<ConstantTextureData*>(tex_data);
    return make_float3(constant->color);
}

// Materials -----------------------------------------------------------------------
extern "C" __device__ void __direct_callable__diffuse(SurfaceInteraction* si, void* mat_data)
{
    const DiffuseData* diffuse = reinterpret_cast<DiffuseData*>(mat_data);
    if (diffuse->twosided)
        si->n = faceforward(si->n, -si->wi, si->n);
    
    si->trace_terminate = false;
    uint32_t seed = si->seed;
    float3 wi = randomSampleHemisphere(seed);
    Onb onb(si->n);
    onb.inverseTransform(wi);
    si->wo = wi;
    si->seed = seed;
    si->albedo = optixDirectCall<float3, const float2&, void*>(diffuse->tex_program_id, si->uv, diffuse->tex_data);
}

extern "C" __device__ void __direct_callable__glass(SurfaceInteraction* si, void* mat_data)
{
    const DielectricData* dielectric = reinterpret_cast<DielectricData*>(mat_data);

    float ni = 1.0f; // air
    float nt = dielectric->ior;  // ior specified 
    float cosine = dot(si->wi, si->n);
    bool into = cosine < 0;
    float3 outward_normal = into ? si->n : -si->n;

    if (!into) swap(ni, nt);

    cosine = fabs(cosine);
    float sine = sqrtf(1.0 - cosine*cosine);
    bool cannot_refract = (ni / nt) * sine > 1.0f;

    float reflect_prob = fresnel(cosine, ni, nt);
    unsigned int seed = si->seed;

    if (cannot_refract || reflect_prob > rnd(seed))
        si->wo = reflect(si->wi, outward_normal);
    else    
        si->wo = refract(si->wi, outward_normal, cosine, ni, nt);
    si->radiance_evaled = false;
    si->trace_terminate = false;
    si->seed = seed;
    si->albedo = optixDirectCall<float3, const float2&, void*>(dielectric->tex_program_id, si->uv, dielectric->tex_data);
}

extern "C" __device__ void __direct_callable__isotropic(SurfaceInteraction* si, void* mat_data)
{
    const IsotropicData* iso = reinterpret_cast<IsotropicData*>(mat_data);
    uint32_t seed = si->seed;
    si->wo = normalize(make_float3(rnd(seed, -1.0f, 1.0f), rnd(seed, -1.0f, 1.0f), rnd(seed, -1.0f, 1.0f)));
    si->trace_terminate = false;
    si->seed = seed;
    si->albedo = iso->albedo;
}

extern "C" __device__ void __direct_callable__area(SurfaceInteraction* si, void* mat_data)
{
    const AreaEmitterData* area = reinterpret_cast<AreaEmitterData*>(mat_data);
    si->trace_terminate = true;
    float is_emitted = dot(si->wi, si->n) < 0.0f ? 1.0f : 0.0f;
    if (area->twosided)
    {
        is_emitted = 1.0f;
        si->n = faceforward(si->n, -si->wi, si->n);
    }

    const float4 base = optixDirectCall<float4, const float2&, void*>(
        area->tex_program_id, si->uv, area->tex_data);
    si->albedo = make_float3(base);
    
    si->emission = si->albedo * area->intensity * is_emitted;
}

