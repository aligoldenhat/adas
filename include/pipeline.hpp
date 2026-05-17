#pragma once
#include "lane_engine.hpp"
#include "video_source.hpp"
#include "yolo_engine.hpp"
#include <memory>
#include <string>

class Pipeline {
  public:
    Pipeline(const std::string &video_path,
             const std::string &yolo_engine_path,
             const std::string &lane_engine_path,
             float              yolo_conf_thresh = 0.4f,
             float              yolo_iou_thresh  = 0.45f,
             float              lane_conf_thresh = 0.1f);
    ~Pipeline();

    void run();

  private:
    std::unique_ptr<YoloEngine> yolo_model_;
    std::unique_ptr<LaneEngine> lane_model_;
    VideoSource                 source_;

    float yolo_conf_thresh_;
    float yolo_iou_thresh_;
    float lane_conf_thresh_;

    // GPU memory for the preprocessed input tensor
    void  *d_input_yolo = nullptr;
    void  *d_input_lane = nullptr;
    float *d_bgr_gpu_   = nullptr; // frame uploaded to GPU

    cudaStream_t stream_ = nullptr;
};
