#include "slicing.h"

namespace ncnn{
DEFINE_LAYER_CREATOR(Slicing)

int Slicing::load_param(const ParamDict& pd)
{
    starts = pd.get(9, Mat());
    ends = pd.get(10, Mat());
    axes = pd.get(11, Mat());
    steps = pd.get(12, Mat());
    return 0;
}

Slicing::Slicing(){
    one_blob_only = true;
    support_inplace = false;
}

int Slicing::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const{
    // assert params are available
    if (starts.w <= 0 || ends.w <= 0 || axes.w <= 0 || steps.w <= 0){
        NCNN_LOGE("Illegal slice parameters");
        return -1;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    // int start, end, axis, step;
    // getSlicingParamsSingle(bottom_blob, start, end, axis, step);
    // NCNN_LOGE("start=%d, end=%d, axis=%d, step=%d", start, end, axis, step);
    NCNN_LOGE("check 1");
    AxisParams sp[3];
    resolveAxisParams(bottom_blob, sp);
    NCNN_LOGE("check 2");
    int out_w = w;//sp[2].toSlice ? sp[2].out_size : w;
    int out_h = sp[1].toSlice ? sp[1].out_size : h;
    int out_c = channels;//sp[0].toSlice ? sp[0].out_size : channels;
    top_blob.create(out_w, out_h, out_c, elemsize, opt.blob_allocator);
    NCNN_LOGE("check 3");
    if (top_blob.empty()){
        return -100;
    }

    // slice channels
    //todo if slice on channel is other than simple crop
    int c_offset = 0;//sp[0].reverse ? sp[0].end : sp[0].start;
    const Mat bottom_blob_sliced = bottom_blob.channel_range(c_offset, out_c);
    NCNN_LOGE("check 4");
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int c = 0; c < out_c; c++){
        const Mat c_src = bottom_blob_sliced.channel(c);
        Mat c_dst = top_blob.channel(c);

        if (elemsize == 1){
            // NCNN_LOGE("Slice mat - 1bit type");
            // slice_mat<signed char>(c_src, c_dst);
        }
        if (elemsize == 2){
            // NCNN_LOGE("Slice mat - int type");
            // slice_mat<unsigned short>(c_src, c_dst);
        }
        if (elemsize == 4){
            NCNN_LOGE("Slice mat - float type");
            slice_mat<float>(c_src, c_dst, sp);
        }
    }
    NCNN_LOGE("check 5");

    // if (axis == 0){ // flip channel axis
    //     NCNN_LOGE("Flip channels axis");
    //     //TODO
    // }else if (axis == 1){ // flip height axis
    //     NCNN_LOGE("Flip heigth axis");
    //     #pragma omp parallel for num_threads(opt.num_threads)
    //     for (int c = 0; c < channels; c++){
    //         const Mat c_src = bottom_blob.channel(c);
    //         Mat c_dst = top_blob.channel(c);

    //         if (elemsize == 1){
    //             NCNN_LOGE("Flip heigth axis - 1bit type");
    //             flip_mat_height_axis<signed char>(bottom_blob, top_blob);
    //         }
    //         if (elemsize == 2){
    //             NCNN_LOGE("Flip heigth axis - int type");
    //             flip_mat_height_axis<unsigned short>(bottom_blob, top_blob);
    //         }
    //         if (elemsize == 4){
    //             NCNN_LOGE("Flip heigth axis - float type");
    //             flip_mat_height_axis<float>(bottom_blob, top_blob);
    //         }
    //     }
        
    // }else if (axis == 2){ // flip width axis
    //     NCNN_LOGE("Flip width axis");
    // }

    return 0;
    
}

void Slicing::resolveAxisParams(const Mat& bottom_blob,  AxisParams* sp) const{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    // size_t elemsize = bottom_blob.elemsize;
    const int* axes_ptr = axes;
    const int* starts_ptr = starts;
    const int* ends_ptr = ends;
    const int* steps_ptr = steps;
    NCNN_LOGE("check a");
    for (int i = 0; i < axes.w; i++){
        NCNN_LOGE("check b");
        int axis = axes_ptr[i]; // 0:c, 1:h, 2:w
        // AxisParams* ap = sp[axis];
        // sp[axis] = new AxisParams();

        sp[axis].toSlice = true;
        sp[axis].reverse = steps_ptr[i] < 0;
        sp[axis].start = starts_ptr[i];
        sp[axis].end = ends_ptr[i];
        sp[axis].step = steps_ptr[i];
        NCNN_LOGE("check c");
        if (sp[axis].end == 1){ // observed for onnx slicing origing from torch.flip(...)
                sp[axis].end = 0;
        }
        NCNN_LOGE("check d");
        if (axis == 0){ //channels
            // if (ap->start == -1){
            //     ap->start = channels -1;
            // }
            // if (ap->end == -1){
            //     ap->end = channels -1;
            // }

        }else if (axis == 1){   //height
            if (sp[axis].start == -1){
                sp[axis].start = h -1;
            }
            if (sp[axis].end == -1){
                sp[axis].end = h -1;
            }
        }else if (axis == 2){   //width
            // if (ap->start == -1){
            //     ap->start = w -1;
            // }
            // if (ap->end == -1){
            //     ap->end = w -1;
            // }
        }
        NCNN_LOGE("check e");

        // resolve out size
        //todo if is odd number
        sp[axis].out_size = ( abs(sp[axis].end - sp[axis].start)+1 ) / abs(sp[axis].step) ;
        NCNN_LOGE("axisParam: [a=%d, sp=%d, st=%d, e=%d, out=%d]", axis, sp[axis].step,
         sp[axis].start, sp[axis].end, sp[axis].out_size);
    }
    
}

void Slicing::getSlicingParamsSingle(const Mat& bottom_blob, int& start, int& end, int& axis, int& step) const{
    const int* starts_ptr = starts;
    const int* ends_ptr = ends;
    const int* axes_ptr = axes;
    const int* steps_ptr = steps;

    start = starts_ptr[0];
    end = ends_ptr[0];
    axis = ends_ptr[0];
    step = steps_ptr[0];

    int flip_axis_size;
    if (axis == 0){
        flip_axis_size = bottom_blob.c;
    }else if (axis == 1){
        flip_axis_size = bottom_blob.h;
    }else{
        flip_axis_size = bottom_blob.w;
    }
    
    if (start == -1){
        start = flip_axis_size -1;
    }

    if (end == 1){
        end = 0;
    }
}

template<typename T>
void Slicing::slice_mat(const Mat& src, Mat& dst, AxisParams* sp) const{
    //todo generalize for all slicing
    NCNN_LOGE("slice mat: %d == %d && %d == %d && %d == -1", src.w, dst.w,
        src.h, dst.h, sp[1].step);
    if (src.w == dst.w && src.h == dst.h && sp[1].step == -1){ // flip height dim case
        flip_mat_height_axis<T>(src, dst);
        return;
    }

    // forward height slicing
    int h_start = sp[1].start;
    int h_end = sp[1].end;
    int h_step = sp[1].step;
    int w = dst.w;
    NCNN_LOGE("slice mat: s=%d, e=%d, sp=%d, w=%d", h_start,h_end,h_step,w);
    const T* ptr = src;
    T* outptr = dst;
    int y_out = 0;
    for (int y = h_start; y <= h_end; y = y + h_step)
    {
        ptr = src.row<T>(y);
        outptr = dst.row<T>(y_out);
        NCNN_LOGE("slicing row %d to out-row %d", y, y_out);
        // todo right only for src.w == dst.w
        if (w < 12){
            for (int x = 0; x < w; x++){
                outptr[x] = ptr[x];
            }
        }
        else{
            memcpy(outptr, ptr, w * sizeof(T));
        }
        y_out++;
    }
}

template<typename T>
void Slicing::flip_mat_height_axis(const Mat& src, Mat& dst){
    int w = dst.w;
    int h = dst.h;
    NCNN_LOGE("Flip function w = %d, h = %d", w, h);
    const T* ptr = src;
    T* outptr = dst;
    int y_src;

    for (int y = 0; y < h; y++)
    {
        y_src = h - y - 1;
        NCNN_LOGE("Flip heigth axis - yin = %d, yout = %d", y_src, y);
        ptr = src.row<T>(y_src);
        outptr = dst.row<T>(y);

        if (w < 12)
        {
            for (int x = 0; x < w; x++)
            {
                outptr[x] = ptr[x];
            }
        }
        else
        {
            memcpy(outptr, ptr, w * sizeof(T));
        }
    }
}
}



