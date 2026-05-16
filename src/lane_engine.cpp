#include "lane_engine.hpp"
#include "NvInferRuntime.h"
#include "model_base.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <fstream>
#include <ios>
#include <iostream>
#include <opencv2/core/types.hpp>
#include <stdexcept>
#include <string>
#include <vector>

class LaneLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} lane_logger;

LaneEngine::~LaneEngine() {
    for (int i = 0; i < 12; i++) {
        if (d_outputs_[i])
            cudaFree(d_outputs_[i]);
    }
    delete ctx_;
    delete engine_;
    delete runtime_;
}

void LaneEngine::load(const std::string &engine_path) {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("Cannot open lane engine");
    size_t size = file.tellg();
    file.seekg(0);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);

    runtime_ = nvinfer1::createInferRuntime(lane_logger);
    engine_  = runtime_->deserializeCudaEngine(buffer.data(), size);
    if (!engine_)
        throw std::runtime_error("Failed to deserialize lane engine");
    ctx_ = engine_->createExecutionContext();
    if (!ctx_)
        throw std::runtime_error("Failed to create lane execution context");

    // Allocate memory for all 12 outputs (3 heads * 4 tensores)
    // Even though we only use Head 2, TRT expects addresses for all bound
    // tensors.
    for (int i = 0; i < 3; i++) {
        cudaMalloc(&d_outputs_[i * 4 + 0],
                   NUM_PRIOR * 2 * sizeof(float)); // logits
        cudaMalloc(&d_outputs_[i * 4 + 1],
                   NUM_PRIOR * 3 * sizeof(float)); // anchors
        cudaMalloc(&d_outputs_[i * 4 + 2],
                   NUM_PRIOR * 1 * sizeof(float)); // lengths
        cudaMalloc(&d_outputs_[i * 4 + 3],
                   NUM_PRIOR * NUM_POINTS * sizeof(float)); // xs
    }
    std::cout << "Lane Engine loaded: " << engine_path << "\n";
}

void LaneEngine::infer_async(void *d_input, cudaStream_t stream) {
    ctx_->setTensorAddress("input", d_input);

    // Bind all 12 outputs using the names which is defined in the Python export
    const char *names[] = {
        "logits_0",
        "anchors_0",
        "lengths_0",
        "xs_0",
        "logits_1",
        "anchors_1",
        "lengths_1",
        "xs_1",
        "logits_2",
        "anchors_2",
        "lengths_2",
        "xs_2",
    };

    for (int i = 0; i < 12; i++) {
        ctx_->setTensorAddress(names[i], d_outputs_[i]);
    }
    ctx_->enqueueV3(stream);
}

std::vector<Lane> LaneEngine::get_lanes(float conf_thresh) {
    // Head 2 (indices 8, 10, 11) is important
    std::vector<float> h_logits(NUM_PRIOR * 2);
    std::vector<float> h_lengths(NUM_PRIOR * 1);
    std::vector<float> h_xs(NUM_PRIOR * NUM_POINTS);

    // Copy Head 2 outputs from GPU to CPU
    cudaMemcpy(h_logits.data(), d_outputs_[8], bytes_logits_, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_lengths.data(), d_outputs_[10], bytes_lengths_, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_xs.data(), d_outputs_[11], bytes_xs_, cudaMemcpyDeviceToHost);

    std::vector<Lane> lanes;
    for (int i = 0; i < NUM_PRIOR; i++) {
        float l0 = h_logits[i * 2 + 0]; // Background logit
        float l1 = h_logits[i * 2 + 1]; // Lane logit

        // Softmax to get probability
        float max_val = std::max(l0, l1);
        float exp0    = std::exp(l0 - max_val);
        float exp1    = std::exp(l1 - max_val);
        float prob    = exp1 / (exp0 + exp1);

        // ADD THIS TEMPORARY DEBUG PRINT:
        if (prob > conf_thresh) {
            std::cout << "Lane candidate! Prob: " << prob << " Length: " << h_lengths[i] << "conf_thresh"
                      << conf_thresh << std::endl;
        }

        if (prob < conf_thresh)
            continue;

        Lane lane;
        lane.score = prob;

        // Length is how many of the 72 points are valid
        int length = std::round(h_lengths[i] * NUM_POINTS);
        length     = std::max(0, std::min(length, NUM_POINTS));

        for (int j = 0; j < length; j++) {
            // Multiply by 800.0f to un-normalize the X coordinate
            float x = h_xs[i * NUM_POINTS + j] * 800.0f;

            // CLRerNet typically usese evely spaced Y anchors.
            // Assuming 72 points spread acrsoss the 320 heights:
            // (Adjust this math if your specific CLRerNet uese different Y-anchors)
            float y = 319.0f - (j * (320.0f / NUM_POINTS));

            lane.points.push_back(cv::Point2f(x, y));
        }

        if (!lane.points.empty()) {
            lanes.push_back(lane);
        }
    }
    return lanes;
}
