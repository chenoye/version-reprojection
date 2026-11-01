#include "solver.hpp"

PnpSolver::PnpSolver() {
    const float ARMOR_WIDTH = 0.135f;
    const float ARMOR_HEIGHT = 0.055f;
    
    // 真实相机内参
    cameraMatrix = (cv::Mat_<double>(3, 3) << 
        2362.279595, 0.0, 679.175139, 
        0.0, 2336.191603, 529.068589, 
        0.0, 0.0, 1.0);
        
    // 真实畸变系数
    distCoeffs = (cv::Mat_<double>(1, 5) << 
        -0.098561, 0.111730, -0.003835, -0.002042, 0.0);

    object_points = {
        cv::Point3f(-ARMOR_WIDTH/2, -ARMOR_HEIGHT/2, 0),
        cv::Point3f(-ARMOR_WIDTH/2,  ARMOR_HEIGHT/2, 0),
        cv::Point3f( ARMOR_WIDTH/2,  ARMOR_HEIGHT/2, 0),
        cv::Point3f( ARMOR_WIDTH/2, -ARMOR_HEIGHT/2, 0) 
    };
}

bool PnpSolver::solve(const std::vector<cv::Point2f>& image_points, cv::Mat& rvec, cv::Mat& tvec) {
    if (image_points.size() != 4) return false;
    return cv::solvePnP(object_points, image_points, cameraMatrix, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
}

bool PnpSolver::solve(const cv::Rect& armor_rect, cv::Mat& rvec, cv::Mat& tvec) {
    std::vector<cv::Point2f> image_points = {
        cv::Point2f(armor_rect.x, armor_rect.y),
        cv::Point2f(armor_rect.x, armor_rect.y + armor_rect.height),
        cv::Point2f(armor_rect.x + armor_rect.width, armor_rect.y + armor_rect.height),
        cv::Point2f(armor_rect.x + armor_rect.width, armor_rect.y)
    };
    return solve(image_points, rvec, tvec);
}