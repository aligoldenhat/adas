#include "pipeline.hpp"
#include "config.hpp"
#include "lane_engine.hpp"
#include "yolo_engine.hpp"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <ratio>
#include <stdexcept>
#include <string>

void launch_preprocess(const uint8_t *d_bgr,
                       float         *d_out,
                       int            src_w,
                       int            src_h,
                       int            dst_w,
                       int            dst_h,
                       cudaStream_t   stream,
                       int            norm_type = 0);

Pipeline::Pipeline(const Config &cfg) : cfg_(cfg) {
    // Open video
    if (!source_.open(cfg.video.source))
        throw std::runtime_error("Cannot open video: " + cfg.video.source);

    // Load YOLO engine
    yolo_model_ = std::make_unique<YoloEngine>();
    yolo_model_->load(cfg.yolo.engine);

    lane_model_ = std::make_unique<LaneEngine>();
    lane_model_->load(cfg.lane.engine);

    cudaStreamCreate(&stream_);

    // Allocate GPU memory for eht raw BGR frame and the preporcessed tensor
    size_t frame_bytes = source_.width() * source_.height() * 3;
    cudaMalloc(&d_bgr_gpu_, frame_bytes);

    // Alllocate seperate input buffers for both models
    auto [yw, yh] = yolo_model_->input_size();
    cudaMalloc(&d_input_yolo, sizeof(float) * 3 * yw * yh);

    auto [lw, lh] = lane_model_->input_size();
    cudaMalloc(&d_input_lane, sizeof(float) * 3 * lw * lh);
}

Pipeline::~Pipeline() {
    cudaFree(d_bgr_gpu_);
    cudaFree(d_input_lane);
    cudaFree(d_input_yolo);
    cudaFree(stream_);
}

void Pipeline::run() {
    auto [yw, yh] = yolo_model_->input_size();
    auto [lw, lh] = lane_model_->input_size();
    cv::Mat frame;

    auto t_prev = std::chrono::steady_clock::now();

    while (true) {
        // Get next frame from video
        if (!source_.next_fram(frame))
            break; // Video ended

        if (!frame.isContinuous())
            frame = frame.clone();

        // Upload fram to GPU
        cudaMemcpyAsync(d_bgr_gpu_,
                        frame.data,
                        frame.cols * frame.rows * frame.elemSize(),
                        cudaMemcpyHostToDevice,
                        stream_);

        // Preprocess on YOLO (640*640)
        launch_preprocess((uint8_t *)d_bgr_gpu_,
                          (float *)d_input_yolo,
                          source_.width(),
                          source_.height(),
                          yw,
                          yh,
                          stream_,
                          0);

        // Preprocess on Lane (800*320)
        launch_preprocess((uint8_t *)d_bgr_gpu_,
                          (float *)d_input_lane,
                          source_.width(),
                          source_.height(),
                          lw,
                          lh,
                          stream_,
                          1);

        // Run both inference on GPU asynchronously
        yolo_model_->infer_async(d_input_yolo, stream_);
        lane_model_->infer_async(d_input_lane, stream_);

        // Wait for GPU to finish
        cudaStreamSynchronize(stream_);

        // Get detections (does NMS inside)
        auto detections = yolo_model_->get_detections(cfg_.yolo.conf_thresh, cfg_.yolo.iou_thresh);
        auto lanes      = lane_model_->get_lanes(cfg_.lane.conf_thresh, cfg_.lane.nms_iou);

        // YOlo scaling math
        float y_scale = std::min(640.0f / source_.width(), 640.0f / source_.height());
        int   y_pad_x = (640 - (int)(source_.width() * y_scale)) / 2;
        int   y_pad_y = (640 - (int)(source_.height() * y_scale)) / 2;

        // Lane scaling math
        float l_scale = std::min(800.0f / source_.width(), 320.0f / source_.height());
        int   l_pad_x = (800 - (int)(source_.width() * l_scale)) / 2;
        int   l_pad_y = (320 - (int)(source_.height() * l_scale)) / 2;

        // FPS calculation
        auto  now = std::chrono::steady_clock::now();
        float ms  = std::chrono::duration<float, std::milli>(now - t_prev).count();
        float fps = 1000.0f / ms;
        t_prev    = now;

        // Draw YOLO
        for (const auto &d : detections) {
            // d.x, d.y, d.x2, d.y2 are in 640x640 space
            // step 1: remove padding
            // step 2: undo scale
            int x1 = (int)((d.x1 - y_pad_x) / y_scale);
            int y1 = (int)((d.y1 - y_pad_y) / y_scale);
            int x2 = (int)((d.x2 - y_pad_x) / y_scale);
            int y2 = (int)((d.y2 - y_pad_y) / y_scale);

            // clamp to frame bounds
            x1 = std::max(0, std::min(x1, frame.cols - 1));
            y1 = std::max(0, std::min(y1, frame.rows - 1));
            x2 = std::max(0, std::min(x2, frame.cols - 1));
            y2 = std::max(0, std::min(y2, frame.rows - 1));

            cv::rectangle(frame, {x1, y1}, {x2, y2}, {0, 255, 0}, 2);
            cv::putText(frame,
                        "cls:" + std::to_string(d.class_id) + " " +
                            std::to_string((int)(d.confidence * 100)) + "%",
                        {x1, y1 - 8},
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.5,
                        {0, 255, 0},
                        1);
        }

        // Draw Lanes
        for (const auto &lane : lanes) {
            std::vector<cv::Point> poly_points;
            for (const auto &pt : lane.points) {
                // Undo letterbox padding and acaling for the lane points
                int orig_x = (int)((pt.x - l_pad_x) / l_scale);
                int orig_y = (int)((pt.y - l_pad_y) / l_scale);
                poly_points.push_back(cv::Point(orig_x, orig_y));
            }
            // Draw the polyline for DEBUG
            if (poly_points.size() > 1)
                std::cout << "Drawing lane with " << poly_points.size() << " points. "
                          << "Start: (" << poly_points.front().x << "," << poly_points.front().y
                          << ") "
                          << "End: (" << poly_points.back().x << "," << poly_points.back().y
                          << ")\n";
            cv::polylines(frame, poly_points, false, cv::Scalar(0, 0, 255), 3);
        }

        cv::putText(frame,
                    "FPS: " + std::to_string((int)fps),
                    {10, 30},
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.0,
                    {0, 200, 255},
                    2);
        cv::imshow(" inference", frame);
        if (cv::waitKey(1) == 'q')
            break;
    }
}
