#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class PnpSolver {
public:
    PnpSolver();
    // 接收4个角点进行解算
    bool solve(const std::vector<cv::Point2f>& image_points, cv::Mat& rvec, cv::Mat& tvec);
    // 兼容旧版矩形输入
    bool solve(const cv::Rect& armor_rect, cv::Mat& rvec, cv::Mat& tvec);

private:
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    std::vector<cv::Point3f> object_points;
};