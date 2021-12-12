#pragma once

#include <prayground/prayground.h>
#include "params.h"

#include "box_medium.h"

using namespace std;

using RaygenRecord = Record<RaygenData>;
using MissRecord = Record<MissData>;
using HitgroupRecord = Record<HitgroupData>;
using EmptyRecord = Record<EmptyData>;

using SBT = ShaderBindingTable<RaygenRecord, MissRecord, HitgroupRecord, EmptyRecord, EmptyRecord, 1>;

class App : public BaseApp 
{
public:
    void setup();
    void update();
    void draw();

    void mousePressed(float x, float y, int button);
    void mouseDragged(float x, float y, int button);
    void mouseReleased(float x, float y, int button);
    void mouseMoved(float x, float y);
    void mouseScrolled(float xoffset, float yoffset);

    void keyPressed(int key);
    void keyReleased(int key);
private:
    void cameraUpdate();
    void initBuffer();

private:
    LaunchParams params; 
    CUDABuffer<LaunchParams> d_params;
    Pipeline pipeline;
    Context context;
    CUstream stream;
    SBT sbt;
    InstanceAccel ias;

    Bitmap result;
    FloatBitmap accum;

    Camera camera;
    bool camera_update;

    EnvironmentEmitter env; 

    float render_time = 0.0f;

    map<string, shared_ptr<Shape>> shapes;
    map<string, shared_ptr<Material>> materials;
    map<string, shared_ptr<Texture>> textures;
    map<string, shared_ptr<AreaEmitter>> lights;
};