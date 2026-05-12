#include "pipeline.hpp"
#include "yolo_engine.hpp"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <ratio>
#include <stdexcept>
#include <string>

void launch_preprocess(const uint8_t *d_bgr, float *d_out, int src_w, int src_h,
                       int dst_w, int dst_h, cudaStream_t stream);

Pipeline::Pipeline(const std::string &video_path,
                   const std::string &engine_path, float conf_thresh,
                   float iou_thresh)
    : conf_thresh_(conf_thresh), iou_thresh_(iou_thresh) {
    // Open video
    if (!source_.open(video_path))
        throw std::runtime_error("Cannot open video: " + video_path);

    // Load YOLO engine
    model_ = std::make_unique<YoloEngine>();
    model_->load(engine_path);

    cudaStreamCreate(&stream_);

    // Allocate GPU memory for eht raw BGR frame and the preporcessed tensor
    size_t frame_bytes = source_.width() * source_.height() * 3;
    cudaMalloc(&d_bgr_gpu_, frame_bytes);

    auto [iw, ih] = model_->input_size();
    cudaMalloc(&d_input_, sizeof(float) * 3 * iw * ih);
}

Pipeline::~Pipeline() {
    cudaFree(d_bgr_gpu_);
    cudaFree(d_input_);
    cudaFree(stream_);
}

void Pipeline::run() {
    auto [iw, ih] = model_->input_size();
    cv::Mat frame;

    auto t_prev = std::chrono::steady_clock::now();

    while (true) {
        // Get next frame from video
        if (!source_.next_fram(frame))
            break; // Video ended

        if (!frame.isContinuous())
            frame = frame.clone();

        // Upload fram to GPU
        cudaMemcpyAsync(d_bgr_gpu_, frame.data,
                        frame.cols * frame.rows * frame.elemSize(),
                        cudaMemcpyHostToDevice, stream_);

        // Preprocess on GPU (resize,, normalize, NCHW)
        launch_preprocess((uint8_t *)d_bgr_gpu_, (float *)d_input_,
                          source_.width(), source_.height(), iw, ih, stream_);

        // Run inference on GPU
        model_->infer_async(d_input_, stream_);

        // Wait for GPU to finish
        cudaStreamSynchronize(stream_);

        // Get detections (does NMS inside)
        auto detections = model_->get_detections(conf_thresh_, iou_thresh_);

        float scale =
            std::min(640.0f / source_.width(), 640.0f / source_.height());
        int new_w = (int)(source_.width() * scale);
        int new_h = (int)(source_.height() * scale);
        int pad_x = (640 - new_w) / 2;
        int pad_y = (640 - new_h) / 2;

        // FPS calculation
        auto  now = std::chrono::steady_clock::now();
        float ms =
            std::chrono::duration<float, std::milli>(now - t_prev).count();
        float fps = 1000.0f / ms;
        t_prev    = now;

        for (const auto &d : detections) {
            // d.x, d.y, d.x2, d.y2 are in 640x640 space
            // step 1: remove padding
            // step 2: undo scale
            int x1 = (int)((d.x1 - pad_x) / scale);
            int y1 = (int)((d.y1 - pad_y) / scale);
            int x2 = (int)((d.x2 - pad_x) / scale);
            int y2 = (int)((d.y2 - pad_y) / scale);

            // clamp to frame bounds
            x1 = std::max(0, std::min(x1, frame.cols - 1));
            y1 = std::max(0, std::min(y1, frame.rows - 1));
            x2 = std::max(0, std::min(x2, frame.cols - 1));
            y2 = std::max(0, std::min(y2, frame.rows - 1));

            cv::rectangle(frame, {x1, y1}, {x2, y2}, {0, 255, 0}, 2);
            cv::putText(frame,
                        "cls:" + std::to_string(d.class_id) + " " +
                            std::to_string((int)(d.confidence * 100)) + "%",
                        {x1, y1 - 8}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        {0, 255, 0}, 1);
        }
        cv::putText(frame, "FPS: " + std::to_string((int)fps), {10, 30},
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 200, 255}, 2);
        cv::imshow("YOLO inference", frame);
        if (cv::waitKey(1) == 'q')
            break;
    }
}
