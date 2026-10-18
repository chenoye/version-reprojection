#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

struct ArmorTarget {
    cv::Rect armor_box;   
    cv::Rect num_box;     
    std::string label;    
    float confidence;     
    double tx, ty, tz;    
};

class ArmorDetector {
public:
    ArmorDetector(const std::string& mlp_path);
    std::vector<ArmorTarget> detect(cv::Mat& img);

private:
    cv::dnn::Net mlp;
    std::vector<std::string> class_names;
};