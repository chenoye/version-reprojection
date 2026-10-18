#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

cv::dnn::Net net;
std::vector<std::string> class_names = {"1", "2", "3", "4", "5", "outpost", "guard", "base"};

void processFrame(cv::Mat& img, int frame_idx, bool is_video) {
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    
    // 1. 全图寻找高亮白块
    cv::Mat mask_digit;
    cv::threshold(gray, mask_digit, 120, 255, cv::THRESH_BINARY);
    
    cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask_digit, mask_digit, cv::MORPH_OPEN, kernel_open);
    cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(mask_digit, mask_digit, cv::MORPH_CLOSE, kernel_close);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask_digit, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 按面积从大到小排序
    std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
        return cv::contourArea(a) > cv::contourArea(b);
    });

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<std::string> box_labels;
    std::vector<cv::Rect> num_boxes;

    // 2. 遍历前 15 个最大的白块
    for (size_t i = 0; i < std::min(contours.size(), (size_t)15); ++i) {
        cv::Rect num_box = cv::boundingRect(contours[i]);

        if (num_box.area() < 100 || num_box.height < 15 || num_box.width < 10) continue;
        float num_ratio = (float)num_box.width / num_box.height;
        if (num_ratio < 0.2 || num_ratio > 1.2) continue;

        // 基于数字反推装甲板大框
        int armor_w = num_box.width * 3.0f;
        int armor_h = num_box.height * 1.5f;
        int cx = num_box.x + num_box.width / 2;
        int cy = num_box.y + num_box.height / 2;
        
        cv::Rect armor_box(cx - armor_w / 2, cy - armor_h / 2, armor_w, armor_h);
        armor_box &= cv::Rect(0, 0, img.cols, img.rows);
        if (armor_box.width <= 0 || armor_box.height <= 0) continue;

        // 验证左右两边是否有红/橙色灯条
        int left_red = 0, right_red = 0;
        for (int y = armor_box.y; y < armor_box.y + armor_box.height; ++y) {
            for (int x = armor_box.x; x < armor_box.x + armor_box.width * 0.25; ++x) {
                cv::Vec3b p = img.at<cv::Vec3b>(y, x);
                if (p[2] > 100 && p[2] > p[0] * 1.1) left_red++; 
            }
            for (int x = armor_box.x + armor_box.width * 0.75; x < armor_box.x + armor_box.width; ++x) {
                cv::Vec3b p = img.at<cv::Vec3b>(y, x);
                if (p[2] > 100 && p[2] > p[0] * 1.1) right_red++;
            }
        }

        if (left_red < 10 || right_red < 10) continue;

        // 送入 MLP 识别
        cv::Mat final_num = mask_digit(num_box).clone();
        if (cv::countNonZero(final_num) > (final_num.total() * 0.5)) cv::bitwise_not(final_num, final_num);
        
        cv::Mat resized_roi;
        cv::resize(final_num, resized_roi, cv::Size(20, 28), 0, 0, cv::INTER_AREA);
        
        cv::Mat float_roi;
        resized_roi.convertTo(float_roi, CV_32F, 1.0 / 255.0);
        cv::Mat blob;
        cv::dnn::blobFromImage(float_roi, blob);
        net.setInput(blob);
        cv::Mat outputs = net.forward();
        
        float max_prob = *std::max_element(outputs.begin<float>(), outputs.end<float>());
        cv::Mat softmax_prob;
        cv::exp(outputs - max_prob, softmax_prob);
        softmax_prob /= cv::sum(softmax_prob)[0];

        cv::Point class_id_point;
        double conf = 0.0;
        cv::minMaxLoc(softmax_prob.reshape(1, 1), nullptr, &conf, nullptr, &class_id_point);
        int label_id = class_id_point.x;

        std::string class_name = "Armor";
        if (conf > 0.15 && label_id >= 0 && label_id < class_names.size()) {
            class_name = class_names[label_id];
            if (is_video && frame_idx % 20 == 0) {
                std::cout << "识别到数字: " << class_name << " (Conf: " << conf * 100 << "%)" << std::endl;
            }
        }

        // ================= 【终极绝杀：非数字直接枪毙】 =================
        // 只有认出了数字，并且置信度达标，才把它放进候选框里
        if (class_name != "Armor" && conf > 0.15) {
            boxes.push_back(armor_box);
            confidences.push_back(static_cast<float>(conf));
            box_labels.push_back(class_name);
            num_boxes.push_back(num_box);
        }
        // ==========================================================
    }

    // 3. NMS 去重
    std::vector<int> indices;
    if (!boxes.empty()) {
        cv::dnn::NMSBoxes(boxes, confidences, 0.1f, 0.4f, indices);
    }

    // 4. 绘制
    for (int idx : indices) {
        cv::Rect armor_box = boxes[idx];
        std::string class_name = box_labels[idx];
        cv::Rect best_num_box = num_boxes[idx];
        float conf = confidences[idx];

        cv::rectangle(img, armor_box, cv::Scalar(0, 255, 0), 3);
        cv::rectangle(img, best_num_box, cv::Scalar(0, 0, 255), 1);
        
        std::stringstream ss;
        ss << class_name << " " << std::fixed << std::setprecision(2) << conf;
        cv::putText(img, ss.str(), cv::Point(armor_box.x, armor_box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    }
}

int main() {
    net = cv::dnn::readNetFromONNX("../assets/mlp.onnx");

    std::cout << "========== 开始处理视频 ==========" << std::endl;
    std::string video_path = "../assets/test_video.mp4"; 
    cv::VideoCapture cap(video_path); 
    if (!cap.isOpened()) return -1;

    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter writer("result_video.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frame_width, frame_height));

    cv::Mat frame;
    int frame_count = 0;
    while (cap.read(frame)) {
        processFrame(frame, frame_count, true);
        writer.write(frame);

        if (frame_count % 60 == 0) {
            std::string debug_name = "debug_frame_" + std::to_string(frame_count) + ".jpg";
            cv::imwrite(debug_name, frame);
            std::cout << ">>> 已自动截图保存: " << debug_name << std::endl;
        }
        frame_count++;
    }
    
    cap.release();
    writer.release();
    std::cout << "视频处理完成！已生成 result_video.mp4" << std::endl;
    return 0;
}