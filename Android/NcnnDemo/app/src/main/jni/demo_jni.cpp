#include <jni.h>
#include <android/log.h>
#include <ncnn/gpu.h>
#include <android/asset_manager_jni.h>
#include <ncnn/net.h>
#include <ncnn/layer.h>

extern "C"{
JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "JNI_OnLoad");

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "JNI_OnUnload");

    ncnn::destroy_gpu_instance();
}
JNIEXPORT void JNICALL
Java_com_lk_ncnndemo_MainActivity_startDemo(JNIEnv *env, jobject thiz,
        jobject asset_manager){
    bool useGpu = false;
    ncnn::Net net;
//    net.register_custom_layer("Slicing", ncnn::layer_creator_func());
    ncnn::Option opt;
    ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
    ncnn::PoolAllocator g_workspace_pool_allocator;

    opt.lightmode = true;
    opt.num_threads = 4;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;

    // use vulkan compute
    if (ncnn::get_gpu_count() != 0)
        opt.use_vulkan_compute = useGpu;

    AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);

    net.opt = opt;

    // init param
    {
        int ret = net.load_param(mgr, "flip.param");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "load_param failed");
            return;
        }
    }

    // init bin
    {
        int ret = net.load_model(mgr, "flip.bin");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "load_model failed");
            return;
        }
    }

    float channel_data[32];
    for (int i = 0; i < 32; ++i) {
        channel_data[i] = 1.f * i;
    }
    ncnn::Mat in = ncnn::Mat(1, 32, 3, sizeof(float));
    for (int i = 0; i < 3; ++i) {
        float * ptr = in.channel(i);
        memcpy(ptr, channel_data,  32 * sizeof(float));
    }

    //test print
    float * ptr = in.channel(1);
    for (int i = 0; i < 32; ++i) {
        __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "input %d = %f",i, ptr[i]);
    }
    // net
    ncnn::Extractor ex = net.create_extractor();

    ex.set_vulkan_compute(useGpu);

    ex.input("input", in);
    ncnn::Mat out;
    ex.extract("output", out);

    __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "out-shape %dx%dx%d", out.c, out.h, out.w);
    float * outptr = out.channel(0);
    for (int i = 0; i < out.h; ++i) {
        __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "output %d = %f", i, outptr[i]);
    }
}
}

