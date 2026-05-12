#include "yolo_engine.hpp"
#include "NvInferRuntime.h"
#include "model_base.hpp"
#include <cstddef>
#include <cuda_runtime.h>
#include <fstream>
#include <ios>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

YoloEngine::~YoloEngine() {
    if (d_output_) {
        cudaFree(d_output_);
    }
    delete ctx_;
    delete engine_;
    delete runtime_;
}

void YoloEngine::load(const std::string &engine_path) {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("Cannot open engine file: " + engine_path);
    size_t size = file.tellg();
    file.seekg(0);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);

    runtime_ = nvinfer1::createInferRuntime(logger);
    engine_  = runtime_->deserializeCudaEngine(buffer.data(), size);

    if (!engine_)
        throw std::runtime_error(
            "Failed to deserialize engine. Version mismatch?");

    ctx_ = engine_->createExecutionContext();

    if (!ctx_)
        throw std::runtime_error("Failed to create TensorRT execution context");

    // YOLO26s output shape is 1*300*6
    output_bytes_ = sizeof(float) * MAX_DETECTIONS * DETECTION_SIZE;
    cudaMalloc(&d_output_, output_bytes_);

    std::cout << "Engine loaded: " << engine_path << "\n";
}

void YoloEngine::infer_async(void *d_input, cudaStream_t stream) {
    ctx_->setTensorAddress("images", d_input);
    ctx_->setTensorAddress("output0", d_output_);
    ctx_->enqueueV3(stream);
}

std::vector<Detection> YoloEngine::get_detections(float conf_thresh,
                                                  float iou_thresh) {
    // copy output from GPU to CPU
    std::vector<float> data(MAX_DETECTIONS * DETECTION_SIZE);
    cudaMemcpy(data.data(), d_output_, output_bytes_, cudaMemcpyDeviceToHost);

    std::vector<Detection> detections;

    // Each column is one candidate detection
    for (int i = 0; i < MAX_DETECTIONS; i++) {

        Detection d;
        // convert corner format (x1, y1, x2, y2) to center format (x, y, w, h)
        // and normalize to 0..1 by dividing by input size 640
        d.x1         = data[i * DETECTION_SIZE + 0];
        d.y1         = data[i * DETECTION_SIZE + 1];
        d.x2         = data[i * DETECTION_SIZE + 2];
        d.y2         = data[i * DETECTION_SIZE + 3];
        d.confidence = data[i * DETECTION_SIZE + 4];
        d.class_id   = (int)data[i * DETECTION_SIZE + 5];

        if (d.confidence < conf_thresh)
            continue;

        detections.push_back(d);
    }

    return detections;
}
