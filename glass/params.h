#pragma once 

#include <optix.h>
#include <prayground/math/vec_math.h>
#include <prayground/math/matrix.h>
#include <prayground/optix/sbt.h>
#include <prayground/core/interaction.h>

using namespace prayground;

struct LaunchParams 
{
    uint32_t width; 
    uint32_t height;
    uint32_t max_depth;
    uint32_t samples_per_launch;
    int frame;
    uchar4* result_buffer;
    float4* accum_buffer;

    OptixTraversableHandle handle;
};

struct CameraData 
{
    float3 origin; 
    float3 lookat;
    float3 U; 
    float3 V; 
    float3 W;
};

struct RaygenData 
{
    CameraData camera;
};

struct HitgroupData 
{
    void* shape_data;
    SurfaceInfo surface_info;
};

struct MissData
{
    void* env_data;
};

struct EmptyData
{

};