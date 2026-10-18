#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class PnpSolver {
public:
    PnpSolver();
    bool solve(const cv::Rect& armor_rect, cv::Mat& rvec, cv::Mat& tvec);

private:
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    std::vector<cv::Point3f> object_points;
};