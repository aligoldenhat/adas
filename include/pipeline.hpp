#pragma once
#include "model_base.hpp"
#include "video_source.hpp"
#include <memory>
#include <string>

class Pipeline {
  public:
    Pipeline(const std::string &video_path, const std::string &engine_path,
             float conf_thresh = 0.4f, float iou_thresh = 0.45f);
    ~Pipeline();

    void run();

  private:
    std::unique_ptr<ModelBase> model_;
    VideoSource                source_;

    float conf_thresh_;
    float iou_thresh_;

    // GPU memory for the preprocessed input tensor
    void  *d_input_   = nullptr;
    float *d_bgr_gpu_ = nullptr; // frame uploaded to GPU

    cudaStream_t stream_ = nullptr;
};
