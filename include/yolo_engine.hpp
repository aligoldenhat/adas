#pragma once
#include "model_base.hpp"
#include <NvInfer.h>
#include <cuda_runtime.h>

class YoloEngine : public ModelBase {
  public:
    YoloEngine() = default;
    ~YoloEngine() override;

    void load(const std::string &engine_path) override;
    void infer_async(void *d_input, cudaStream_t stream) override;
    std::vector<Detection> get_detections(float conf_thresh,
                                          float iou_thresh);

    std::string name() const override {
        return "yolov8";
    }
    std::pair<int, int> input_size() const override {
        return {640, 640};
    }

  private:
    // TensorRT objects that manage the engine lifetime
    nvinfer1::IRuntime          *runtime_ = nullptr;
    nvinfer1::ICudaEngine       *engine_  = nullptr;
    nvinfer1::IExecutionContext *ctx_     = nullptr;

    // GPU memory buffers for input and output tensors
    void  *d_output_     = nullptr;
    size_t output_bytes_ = 0;

    // 1*300*6 - 300 detections, each has [x1, y1, x2, y2, conf, class_id]
    static constexpr int MAX_DETECTIONS = 300;
    static constexpr int DETECTION_SIZE = 6;
};
