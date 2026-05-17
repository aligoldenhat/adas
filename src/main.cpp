#include "model_base.hpp"
#include "pipeline.hpp"
#include "yolo_engine.hpp"
#include <exception>
#include <iostream>
#include <memory>
#include <string>

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./yolo_run <video_path> <engine_path>\n";
        std::cerr << "Example: ./yolo_run /tmp/video.mp4 yolov8n.engine\n";
        return 1;
    }

    std::string video_path       = argv[1];
    std::string yolo_engine_path = argv[2];
    std::string lane_engine_path = argv[3];

    ModelRegistry::get().register_model("yolo26s", [] { return std::make_unique<YoloEngine>(); });

    try {
        Pipeline pipeline(video_path, yolo_engine_path, lane_engine_path, 0.4f, 0.45f, 0.04f);
        pipeline.run();
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
