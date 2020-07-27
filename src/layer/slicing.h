#ifndef LAYER_SLICING_H
#define LAYER_SLICING_H
#include "layer.h"

namespace ncnn {

struct AxisParams
{
    int start = 0;
    int end = 0;
    int step = 0;
    int out_size = 0;
    bool reverse = false;
    bool toSlice = false;
};

class Slicing : public Layer
{
public:
    Slicing();
    virtual int load_param(const ParamDict& pd);
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    void getSlicingParams(std::vector<int>& starts, std::vector<int>& ends, std::vector<int>& axes,
     std::vector<int>& steps) const;
    void getSlicingParamsSingle(const Mat& bottom_blob, int& start, int& end, int& axis, int& step) const;

    void resolveAxisParams(const Mat& bottom_blob, AxisParams* sp) const;

    template<typename T>
    static void flip_mat_height_axis(const Mat& src, Mat& dst);

    template<typename T>
    void slice_mat(const Mat& src, Mat& dst, AxisParams* sp) const;

    Mat starts;
    Mat ends;
    Mat axes;
    Mat steps;

    // AxisParams sp[3]; // 0:c, 1:h, 2:w
};
}

#endif // LAYER_SLICING_H
