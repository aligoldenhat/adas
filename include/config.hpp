#pragma once
#include "nlohmann/detail/macro_scope.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <string>

// sub-configs with INTRUSIVE macros
struct VideoConfig {
    std::string source;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(VideoConfig, source)
};

struct YoloConfig {
    std::string engine;
    float       conf_thresh = 0.4f;
    float       iou_thresh  = 0.45f;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(YoloConfig, engine, conf_thresh, iou_thresh)
};

struct LaneConfig {
    std::string engine;
    float       conf_thresh = 0.04f;
    float       nms_iou     = 0.50f;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(LaneConfig, engine, conf_thresh, nms_iou)
};

// Main Config
struct Config {
    VideoConfig video;
    YoloConfig  yolo;
    LaneConfig  lane;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Config, video, yolo, lane)

    static Config load(const std::string &path) {
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("Cannot open config: " + path);

        nlohmann::json j;
        f >> j;

        return j.get<Config>();
    }
};
