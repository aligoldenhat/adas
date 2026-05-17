#pragma once
#include "NvInferRuntime.h"
#include "model_base.hpp"
#include <NvInfer.h>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

class LaneEngine : public ModelBase {
  public:
    LaneEngine() = default;
    ~LaneEngine();

    void load(const std::string &engine_path) override;
    void infer_async(void *d_input, cudaStream_t stream) override;

    std::vector<Lane> get_lanes(float conf_thresh);
    std::vector<Lane> nms_lanes(std::vector<Lane> &candidates, float x_distance_thresh);

        std::string name() const override {
        return "clrernet";
    }
    std::pair<int, int> input_size() const override {
        return {800, 320};
    };

  private:
    nvinfer1::IRuntime          *runtime_ = nullptr;
    nvinfer1::ICudaEngine       *engine_  = nullptr;
    nvinfer1::IExecutionContext *ctx_     = nullptr;

    // We have 12 outputs. we'll store their device pointers here.
    void *d_outputs_[12] = {nullptr};

    static constexpr int NUM_PRIOR  = 192;
    static constexpr int NUM_POINTS = 72;

    // Sizes in bytes for Head 2
    size_t bytes_logits_  = NUM_PRIOR * 2 * sizeof(float);
    size_t bytes_lengths_ = NUM_PRIOR * 1 * sizeof(float);
    size_t bytes_xs_      = NUM_PRIOR * NUM_POINTS * sizeof(float);
};
