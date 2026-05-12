#pragma once
#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

class ModelBase {
  public:
    virtual ~ModelBase() = default;
    virtual void load(const std::string &engine_path) = 0;
    virtual void infer_async(void *d_input, cudaStream_t stream) = 0;
    virtual std::vector<Detection> get_detections(float conf_thresh,
                                                  float iou_thresh) = 0;
    virtual std::string name() const = 0;
    virtual std::pair<int, int> input_size() const = 0;
};

class ModelRegistry {
  public:
    using Factory = std::function<std::unique_ptr<ModelBase>()>;

    static ModelRegistry &get() {
        static ModelRegistry instance;
        return instance;
    }

    void register_model(const std::string &name, Factory f) {
        factories_[name] = f;
    }

    std::unique_ptr<ModelBase> create(const std::string &name) {
        return factories_.at(name)();
    }

  private:
    std::unordered_map<std::string, Factory> factories_;
};
