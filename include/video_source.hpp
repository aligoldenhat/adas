#pragma once
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <string>

class VideoSource {
  public:
    // Open a local file or an already-resolved stream URL
    bool open(const std::string &path);

    // Grab the next frame - returns flase when the video ends
    bool next_fram(cv::Mat &frame);

    int width() const {
        return width_;
    }
    int height() const {
        return height_;
    }

  private:
    cv::VideoCapture cap_;
    int width_  = 0;
    int height_ = 0;
};
