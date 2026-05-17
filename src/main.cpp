#include "config.hpp"
#include "model_base.hpp"
#include "pipeline.hpp"
#include "yolo_engine.hpp"
#include <exception>
#include <iostream>
#include <memory>
#include <string>

int main(int argc, char *argv[]) {
    std::string config_path = "config.json";

    // Providing the config.json file is allowable
    if (argc >= 2) {
        config_path = argv[1];
    }

    Config cfg = Config::load(config_path);

    ModelRegistry::get().register_model("yolo26s", [] { return std::make_unique<YoloEngine>(); });

    try {
        Pipeline pipeline(cfg);
        pipeline.run();
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
