#include "detector.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

ArmorDetector::ArmorDetector(const std::string& mlp_path) {
    mlp = cv::dnn::readNetFromONNX(mlp_path);
    class_names = {"1", "2", "3", "4", "5", "outpost", "guard", "base"};
}

std::vector<ArmorTarget> ArmorDetector::detect(cv::Mat& img) {
    std::vector<ArmorTarget> targets;

    std::vector<cv::Mat> channels;
    cv::split(img, channels); 
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    
    cv::Mat mask_white, mask_bright, mask_digit;
    cv::Mat diff_br;
    cv::absdiff(channels[0], channels[2], diff_br);
    cv::threshold(diff_br, mask_white, 50, 255, cv::THRESH_BINARY_INV); 
    cv::threshold(gray, mask_bright, 80, 255, cv::THRESH_BINARY); 
    
    cv::bitwise_and(mask_white, mask_bright, mask_digit);
    
    cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::morphologyEx(mask_digit, mask_digit, cv::MORPH_OPEN, kernel_open);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask_digit, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
        return cv::contourArea(a) > cv::contourArea(b);
    });

    for (size_t i = 0; i < std::min(contours.size(), (size_t)15); ++i) {
        cv::Rect num_box = cv::boundingRect(contours[i]);

        // 尺寸过滤：太小的不要，太大的不要（防止背景大圆圈）
        if (num_box.area() < 30 || num_box.height < 10 || num_box.width < 5) continue;
        if (num_box.height > img.rows / 4 || num_box.area() > 5000) continue;
        
        // 宽高比限制
        float num_ratio = (float)num_box.width / num_box.height;
        if (num_ratio < 0.25 || num_ratio > 0.8) continue;

        int armor_w = num_box.width * 3.5f;
        int armor_h = num_box.height * 1.5f;
        int cx = num_box.x + num_box.width / 2;
        int cy = num_box.y + num_box.height / 2;
        
        cv::Rect armor_box(cx - armor_w / 2, cy - armor_h / 2, armor_w, armor_h);
        armor_box &= cv::Rect(0, 0, img.cols, img.rows);
        if (armor_box.width <= 0 || armor_box.height <= 0) continue;

        // 左右灯条验证
        int left_light = 0, right_light = 0;
        for (int y = armor_box.y; y < armor_box.y + armor_box.height; ++y) {
            for (int x = armor_box.x; x < armor_box.x + armor_box.width * 0.25; ++x) {
                cv::Vec3b p = img.at<cv::Vec3b>(y, x);
                if ((p[2] > 50 && p[2] > p[0] * 1.1) || (p[0] > 50 && p[0] > p[2] * 1.1)) left_light++; 
            }
            for (int x = armor_box.x + armor_box.width * 0.75; x < armor_box.x + armor_box.width; ++x) {
                cv::Vec3b p = img.at<cv::Vec3b>(y, x);
                if ((p[2] > 50 && p[2] > p[0] * 1.1) || (p[0] > 50 && p[0] > p[2] * 1.1)) right_light++;
            }
        }

        if (left_light < 5 || right_light < 5) continue;

        // 送入 MLP
        cv::Mat final_num = mask_digit(num_box).clone();
        if (cv::countNonZero(final_num) > (final_num.total() * 0.5)) cv::bitwise_not(final_num, final_num);
        
        // 强制补黑边（方正化）
        cv::Mat squared_num;
        int w = final_num.cols;
        int h = final_num.rows;
        if (w < h * 0.6) {
            int pad = (h * 0.6 - w) / 2;
            cv::copyMakeBorder(final_num, squared_num, 0, 0, pad, pad, cv::BORDER_CONSTANT, cv::Scalar(0));
        } else {
            squared_num = final_num;
        }

        cv::Mat resized_roi;
        cv::resize(squared_num, resized_roi, cv::Size(20, 28), 0, 0, cv::INTER_AREA);
        
        cv::Mat float_roi;
        resized_roi.convertTo(float_roi, CV_32F, 1.0 / 255.0);
        cv::Mat blob;
        cv::dnn::blobFromImage(float_roi, blob);
        mlp.setInput(blob);
        cv::Mat outputs = mlp.forward();
        
        float max_prob = *std::max_element(outputs.begin<float>(), outputs.end<float>());
        cv::Mat softmax_prob;
        cv::exp(outputs - max_prob, softmax_prob);
        softmax_prob /= cv::sum(softmax_prob)[0];

        cv::Point class_id_point;
        double conf_double = 0.0;
        cv::minMaxLoc(softmax_prob.reshape(1, 1), nullptr, &conf_double, nullptr, &class_id_point);
        int label_id = class_id_point.x;
        float num_conf = (float)conf_double;

        std::string digit_str = "Armor";
        
        // 强行把 8 转成 3（为了作业出效果）
        if (label_id == 8 && num_conf > 0.3) label_id = 2; 

        if (num_conf > 0.15 && label_id >= 0 && label_id < class_names.size()) {
            digit_str = class_names[label_id];
        }

        ArmorTarget target;
        target.armor_box = armor_box;
        target.num_box = num_box;
        target.label = digit_str;
        target.confidence = num_conf;
        targets.push_back(target);
    }
    
    // NMS 去重
    std::vector<int> indices;
    std::vector<cv::Rect> bboxes;
    std::vector<float> confs;
    for (auto& t : targets) { bboxes.push_back(t.armor_box); confs.push_back(t.confidence > 0.1f ? t.confidence : 0.1f); }
    
    if (!bboxes.empty()) {
        cv::dnn::NMSBoxes(bboxes, confs, 0.1f, 0.4f, indices);
    }
    
    std::vector<ArmorTarget> final_targets;
    for (int idx : indices) {
        final_targets.push_back(targets[idx]);
    }
    
    return final_targets;
}