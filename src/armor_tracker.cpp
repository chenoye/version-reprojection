#include "armor_tracker.hpp"
#include <sstream>
#include <cmath>

ArmorTracker::ArmorTracker() : ekf(8, 4, 0, CV_32F) {
    cv::setIdentity(ekf.transitionMatrix);
    ekf.transitionMatrix.at<float>(0, 4) = dt; 
    ekf.transitionMatrix.at<float>(1, 5) = dt; 
    ekf.transitionMatrix.at<float>(2, 6) = dt; 
    ekf.transitionMatrix.at<float>(3, 7) = dt; 

    ekf.measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);
    ekf.measurementMatrix.at<float>(0, 0) = 1.0f;
    ekf.measurementMatrix.at<float>(1, 1) = 1.0f;
    ekf.measurementMatrix.at<float>(2, 2) = 1.0f;
    ekf.measurementMatrix.at<float>(3, 3) = 1.0f;

    cv::setIdentity(ekf.processNoiseCov, cv::Scalar::all(1e-2));
    ekf.measurementNoiseCov = (cv::Mat_<float>(4, 4) << 
        1e-2, 0, 0, 0,
        0, 1e-2, 0, 0,
        0, 0, 1e-1, 0,
        0, 0, 0, 5e-1); 
    cv::setIdentity(ekf.errorCovPost, cv::Scalar::all(1));

    csv_file.open("ekf_data.csv");
    // 写入表头
    csv_file << "cx,cy,cz,yaw,vx,vy,vz,omega\n";
}

std::vector<ImuData> ArmorTracker::loadImuData(const std::string& path) {
    std::vector<ImuData> data;
    std::ifstream file(path);
    if (!file.is_open()) return data;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        ImuData imu;
        if (ss >> imu.timestamp >> imu.w >> imu.x >> imu.y >> imu.z) {
            data.push_back(imu);
        }
    }
    return data;
}

cv::Mat ArmorTracker::predictAndUpdate(const cv::Mat& pnp_tvec, float board_yaw, float imu_yaw) {
    ekf.predict();

    float yaw = ekf.statePre.at<float>(3);
    ekf.measurementMatrix.at<float>(0, 3) = -radius * std::sin(yaw);
    ekf.measurementMatrix.at<float>(1, 3) =  radius * std::cos(yaw);

    float rx = pnp_tvec.at<double>(0);
    float ry = pnp_tvec.at<double>(1);
    float rz = pnp_tvec.at<double>(2);

    float world_x = rx * std::cos(imu_yaw) - ry * std::sin(imu_yaw);
    float world_y = rx * std::sin(imu_yaw) + ry * std::cos(imu_yaw);

    cv::Mat measurement = (cv::Mat_<float>(4, 1) << world_x, world_y, rz, board_yaw);
    return ekf.correct(measurement);
}

void ArmorTracker::saveDataToCsv(const cv::Mat& state) {
    if (csv_file.is_open()) {
        csv_file << state.at<float>(0) << "," << state.at<float>(1) << "," 
                 << state.at<float>(2) << "," << state.at<float>(3) << ","
                 << state.at<float>(4) << "," << state.at<float>(5) << ","
                 << state.at<float>(6) << "," << state.at<float>(7) << "\n";
    }
}