#ifndef LAYER_SLICING_VULKAN_H
#define LAYER_SLICING_VULKAN_H
#include "slicing.h"

namespace ncnn{
class Slicing_vulkan : virtual public Slicing{
public:
    Slicing_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

public:
    Pipeline* pipeline;
    Pipeline* pipeline_pack4;
    Pipeline* pipeline_pack1to4;
    Pipeline* pipeline_pack4to1;
    Pipeline* pipeline_pack8;
    Pipeline* pipeline_pack1to8;
    Pipeline* pipeline_pack4to8;
    Pipeline* pipeline_pack8to4;
    Pipeline* pipeline_pack8to1;

};
}

#endif