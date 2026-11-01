#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>

struct ImuData {
    double timestamp;
    double w, x, y, z; 
};

class ArmorTracker {
public:
    ArmorTracker();
    std::vector<ImuData> loadImuData(const std::string& path);
    cv::Mat predictAndUpdate(const cv::Mat& pnp_tvec, float board_yaw, float imu_yaw);
    void saveDataToCsv(const cv::Mat& state);

private:
    cv::KalmanFilter ekf;
    int stateSize = 8; // [cx, cy, cz, yaw, vx, vy, vz, omega]
    int measSize = 4;  // [x, y, z, yaw]
    float dt = 0.03f;  
    float radius = 0.2f; 
    std::ofstream csv_file;
};