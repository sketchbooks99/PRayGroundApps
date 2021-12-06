#include "app.h"

// ------------------------------------------------------------------
void App::setup()
{
    stream = 0; 
    CUDA_CHECK(cudaFree(0));

    OPTIX_CHECK(optixInit());
    context.disableValidation();
    context.create();

    ias = InstanceAccel(InstanceAccel::Type::Instances);

    pipeline.setLaunchVariableName("params");
    pipeline.setDirectCallableDepth(5);
    pipeline.setContinuationCallableDepth(5);
    pipeline.setNumAttributes(5);
    pipeline.setNumPayloads(5);

    Module module = pipeline.createModuleFromCudaFile(context, "kernels.cu");
    module.setDebugLevel(OPTIX_COMPILE_DEBUG_LEVEL_MODERATE);

    result.allocate(PixelFormat::RGBA, pgGetWidth(), pgGetHeight());
    accum.allocate(PixelFormat::RGBA, pgGetWidth(), pgGetHeight());

    params.width = result.width();
    params.height = result.height();
    params.samples_per_launch = 1;
    params.max_depth = 10;
    params.frame = 0;
    initBuffer();

    camera.setOrigin(0, -5.0f, 55);
    camera.setLookat(0, -5.0f, 0);
    camera.setUp(0, 1, 0);
    camera.setFov(40.0f);
    camera.setAspect((float)params.width / params.height);
    camera.enableTracking(pgGetCurrentWindow());
    float3 U, V, W;
    camera.UVWFrame(U, V, W);

    ProgramGroup raygen = pipeline.createRaygenProgram(context, module, "__raygen__pinhole");
    RaygenRecord rg_record;
    raygen.recordPackHeader(&rg_record);
    rg_record.data.camera = 
    {
        .origin = camera.origin(), 
        .lookat = camera.lookat(), 
        .U = U, 
        .V = V, 
        .W = W
    };
    sbt.setRaygenRecord(rg_record);

    auto setupCallable = [&](const string& dc, const string& cc) -> uint32_t
    {
        EmptyRecord record = {};
        auto [prg, id] = pipeline.createCallablesProgram(context, module, dc, cc);
        prg.recordPackHeader(&record);
        sbt.addCallablesRecord(record);
        return id;
    };

    uint32_t constant_prg_id = setupCallable(DC_FUNC_STR("constant"), "");

    uint32_t diffuse_prg_id = setupCallable(DC_FUNC_STR("diffuse"), "");
    uint32_t glass_prg_id = setupCallable(DC_FUNC_STR("glass"), "");
    uint32_t area_prg_id = setupCallable(DC_FUNC_STR("area"), "");

    textures.emplace("env", new ConstantTexture(make_float3(0.0f), constant_prg_id));

    env = EnvironmentEmitter(textures.at("env"));
    env.copyToDevice();

    ProgramGroup miss = pipeline.createMissProgram(context, module, MS_FUNC_STR("envmap"));
    MissRecord ms_record;
    miss.recordPackHeader(&ms_record);
    ms_record.data.env_data = env.devicePtr();
    sbt.setMissRecord(ms_record);

    ProgramGroup plane_prg = pipeline.createHitgroupProgram(context, module, CH_FUNC_STR("plane"), IS_FUNC_STR("plane"));
    ProgramGroup sphere_prg = pipeline.createHitgroupProgram(context, module, CH_FUNC_STR("sphere"), IS_FUNC_STR("sphere"));
    ProgramGroup mesh_prg = pipeline.createHitgroupProgram(context, module, CH_FUNC_STR("mesh"));

    struct Primitive
    {
        shared_ptr<Shape> shape;
        shared_ptr<Material> material;
        uint32_t sample_bsdf_id;
        uint32_t pdf_id;
    };

    uint32_t sbt_idx = 0;
    uint32_t sbt_offset = 0; 
    uint32_t instance_id = 0;

    using SurfaceP = variant<shared_ptr<Material>, shared_ptr<AreaEmitter>>;
    auto addHitgroupRecord = [&](ProgramGroup& prg, shared_ptr<Shape> shape, SurfaceP surface, uint32_t sample_bsdf_id, uint32_t pdf_id)
    {
        const bool is_mat = holds_alternative<shared_ptr<Material>>(surface);

        // Copy data to GPU
        shape->copyToDevice();
        shape->setSbtIndex(sbt_idx);
        if (is_mat) std::get<shared_ptr<Material>>(surface)->copyToDevice();
        else        std::get<shared_ptr<AreaEmitter>>(surface)->copyToDevice();

        // Register data to shader binding table
        HitgroupRecord record;
        prg.recordPackHeader(&record);
        record.data =
        {
            .shape_data = shape->devicePtr(),
            .surface_info =
            {
                .data = is_mat ? std::get<shared_ptr<Material>>(surface)->devicePtr() : std::get<shared_ptr<AreaEmitter>>(surface)->devicePtr(),
                .sample_id = sample_bsdf_id,
                .bsdf_id = sample_bsdf_id,
                .pdf_id = pdf_id,
                .type = is_mat ? std::get<shared_ptr<Material>>(surface)->surfaceType() : SurfaceType::AreaEmitter,
            }
        };

        sbt.addHitgroupRecord(record);
        sbt_idx++;
    };

    auto createGAS = [&](const shared_ptr<Shape>& shape, const Matrix4f& transform) 
    {
        ShapeInstance instance{shape->type(), shape, transform};
        instance.allowCompaction();
        instance.buildAccel(context, stream);
        instance.setSBTOffset(sbt_offset);
        instance.setId(instance_id);

        ias.addInstance(instance);

        instance_id++;
        sbt_offset += SBT::NRay;
    };

    auto setupPrimitive = [&](ProgramGroup& prg, const Primitive& p, const Matrix4f& transform)
    {
        addHitgroupRecord(prg, p.shape, p.material, p.sample_bsdf_id, p.pdf_id);
        createGAS(p.shape, transform);
    };

    auto setupAreaEmitter = [&](ProgramGroup& prg, const shared_ptr<Shape>& shape, const shared_ptr<AreaEmitter>& area, const Matrix4f& transform)
    {
        addHitgroupRecord(prg, shape, area, area_prg_id, area_prg_id);
        createGAS(shape, transform);
    };

    textures.emplace("red", new ConstantTexture(make_float3(0.8f, 0.05f, 0.05f), constant_prg_id));
    textures.emplace("green", new ConstantTexture(make_float3(0.05f, 0.8f, 0.05f), constant_prg_id));
    textures.emplace("white", new ConstantTexture(make_float3(1.0f), constant_prg_id));
    textures.emplace("wall", new ConstantTexture(make_float3(0.8f), constant_prg_id));
    textures.emplace("ceiling", new ConstantTexture(make_float3(0.9f, 0.8f, 0.5f), constant_prg_id));

    materials.emplace("wall", new Diffuse(textures.at("wall")));
    materials.emplace("red", new Diffuse(textures.at("red")));
    materials.emplace("green", new Diffuse(textures.at("green")));
    materials.emplace("glass", new Dielectric(textures.at("white"), 1.5f));

    shapes.emplace("plane", new Plane(make_float2(-1.0f), make_float2(1.0f)));
    auto bunny = make_shared<TriangleMesh>(pgPathJoin(pgRootDir(), "resources/model/bunny.obj"));
    bunny->smooth();
    shapes.emplace("bunny", bunny);
    shapes.emplace("sphere", new Sphere(make_float3(0.0f), 1.0f));

    lights.emplace("ceiling", new AreaEmitter(textures.at("ceiling"), 20.0f, true));

    constexpr float size = 20.0f;

    Primitive left{shapes.at("plane"), materials.at("red"), diffuse_prg_id, diffuse_prg_id};
    setupPrimitive(plane_prg, left, Matrix4f::translate(-size, 0, 0) * Matrix4f::rotate(math::pi/2.0f, {0.0f, 0.0f, 1.0f}) * Matrix4f::scale(size));

    Primitive right{shapes.at("plane"), materials.at("green"), diffuse_prg_id, diffuse_prg_id};
    setupPrimitive(plane_prg, right, Matrix4f::translate(size, 0, 0) * Matrix4f::rotate(math::pi/2.0f, {0.0f, 0.0f, 1.0f}) * Matrix4f::scale(size));

    Primitive ceiling{shapes.at("plane"), materials.at("wall"), diffuse_prg_id, diffuse_prg_id};
    setupPrimitive(plane_prg, ceiling, Matrix4f::translate(0, size, 0) * Matrix4f::scale(size));

    Primitive floor{shapes.at("plane"), materials.at("wall"), diffuse_prg_id, diffuse_prg_id};
    setupPrimitive(plane_prg, floor, Matrix4f::translate(0, -size, 0) * Matrix4f::scale(size));

    Primitive back{shapes.at("plane"), materials.at("wall"), diffuse_prg_id, diffuse_prg_id};
    setupPrimitive(plane_prg, back, Matrix4f::translate(0, 0, -size) * Matrix4f::rotate(math::pi/2.0f, {1.0f, 0.0f, 0.0f}) * Matrix4f::scale(size));

    // Primitive glass{shapes.at("bunny"), materials.at("glass"), glass_prg_id, glass_prg_id};
    // setupPrimitive(mesh_prg, glass, Matrix4f::translate(0.0f, -25.0f, 0.0f) * Matrix4f::scale(150.0f));
    Primitive glass{shapes.at("sphere"), materials.at("glass"), glass_prg_id, glass_prg_id};
    setupPrimitive(sphere_prg, glass, Matrix4f::translate(0.0f, -10.0f, 0.0f) * Matrix4f::scale(10.0f));

    setupAreaEmitter(plane_prg, shapes.at("plane"), lights.at("ceiling"), Matrix4f::translate(0, size - 0.5f, 0) * Matrix4f::scale(size / 4));

    CUDA_CHECK(cudaStreamCreate(&stream));
    ias.build(context, stream);
    sbt.createOnDevice();
    params.handle = ias.handle();
    pipeline.create(context);
    d_params.allocate(sizeof(LaunchParams));
}

// ------------------------------------------------------------------
void App::update()
{
    cameraUpdate();

    d_params.copyToDeviceAsync(&params, sizeof(LaunchParams), stream);

    optixLaunch(
        static_cast<OptixPipeline>(pipeline), 
        stream, 
        d_params.devicePtr(), 
        sizeof(LaunchParams), 
        &sbt.sbt(), 
        params.width, 
        params.height, 
        1
    );

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_SYNC_CHECK();

    params.frame += params.samples_per_launch;
    result.copyFromDevice();
}

// ------------------------------------------------------------------
void App::draw()
{
    result.draw(0, 0);
    if (params.frame == 5000) {
        result.write(pgPathJoin(pgAppDir(), "sphere.jpg"));
        pgExit();
    }
}

// ------------------------------------------------------------------
void App::close()
{

}

// ------------------------------------------------------------------
void App::mousePressed(float x, float y, int button)
{
    
}

// ------------------------------------------------------------------
void App::mouseDragged(float x, float y, int button)
{
    if (button != MouseButton::Middle) return;
    camera_update = true;
}

// ------------------------------------------------------------------
void App::mouseReleased(float x, float y, int button)
{
    
}

// ------------------------------------------------------------------
void App::mouseMoved(float x, float y)
{
    
}

// ------------------------------------------------------------------
void App::mouseScrolled(float x, float y)
{
    camera_update = true;
}

// ------------------------------------------------------------------
void App::keyPressed(int key)
{

}

// ------------------------------------------------------------------
void App::keyReleased(int key)
{

}

// ------------------------------------------------------------------
void App::cameraUpdate()
{
    if (!camera_update)
        return;
    camera_update = false;

    float3 U, V, W;
    camera.UVWFrame(U, V, W);

    RaygenRecord* rg_record = reinterpret_cast<RaygenRecord*>(sbt.raygenRecord());
    RaygenData rg_data;
    rg_data.camera =
    {
        .origin = camera.origin(),
        .lookat = camera.lookat(),
        .U = U,
        .V = V,
        .W = W
    };

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(&rg_record->data),
        &rg_data, sizeof(RaygenData),
        cudaMemcpyHostToDevice
    ));

    initBuffer();
}

// ------------------------------------------------------------------
void App::initBuffer()
{
    params.frame = 0;

    result.allocateDevicePtr();
    accum.allocateDevicePtr();

    params.result_buffer = (uchar4*)result.devicePtr();
    params.accum_buffer = (float4*)accum.devicePtr();

    CUDA_SYNC_CHECK();
}