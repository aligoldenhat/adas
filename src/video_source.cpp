#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <stdexcept>
#include <string>
#include <video_source.hpp>

bool VideoSource::open(const std::string &path) {
    // Path can be:
    //  - a local file : "/tmp/video.mp4"
    //  - a direct URL
    //  - an RSTP stream
    cap_.open(path);

    if (!cap_.isOpened()) {
        std::cerr << "Failed to open video: " << path << "\n";
        return false;
    }

    width_  = (int)cap_.get(cv::CAP_PROP_FRAME_WIDTH);
    height_ = (int)cap_.get(cv::CAP_PROP_FRAME_HEIGHT);

    std::cout << "Opened video: " << width_ << "x" << height_ << "\n";
    return true;
}

bool VideoSource::next_fram(cv::Mat &frame) {
    return cap_.read(frame);
}
