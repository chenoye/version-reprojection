#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "solver.hpp" // 引入我们写好的 PnP 模块

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// YOLO 的图片批处理与解算函数
void processImage(const std::string& img_path, Ort::Session& session, 
                  const char* input_names[], const char* output_names[],
                  const std::vector<std::string>& class_names, PnpSolver& solver, int img_idx) {
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) return;

    std::cout << "\n--- YOLO 处理: " << img_path << " ---" << std::endl;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(640, 480));
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32F, 1.0 / 255.0); 

    std::vector<float> input_data(3 * 480 * 640);
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels); 
    for (int c = 0; c < 3; ++c) {
        std::memcpy(input_data.data() + c * 480 * 640, channels[c].data, 480 * 640 * sizeof(float));
    }

    std::vector<int64_t> input_shape = {1, 3, 480, 640};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    float* floatarr = output_tensors[0].GetTensorMutableData<float>();

    int num_anchors = 6300;
    float scale_x = (float)img.cols / 640.0f;
    float scale_y = (float)img.rows / 480.0f;

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    for (int i = 0; i < num_anchors; ++i) {
        float max_class_conf = 0.0f;
        int best_class_id = -1;
        for (int c = 4; c <= 12; ++c) {
            float score = sigmoid(floatarr[c * num_anchors + i]);
            if (score > max_class_conf) {
                max_class_conf = score;
                best_class_id = c - 4;
            }
        }

        if (max_class_conf > 0.6f) {
            std::vector<cv::Point> pts;
            for (int k = 0; k < 4; ++k) {
                float raw_x = floatarr[(13 + k * 2) * num_anchors + i];
                float raw_y = floatarr[(14 + k * 2) * num_anchors + i];
                int px = static_cast<int>(raw_x * scale_x);
                int py = static_cast<int>(raw_y * scale_y);
                pts.push_back(cv::Point(px, py));
            }

            cv::Rect armor_box = cv::boundingRect(pts);
            armor_box &= cv::Rect(0, 0, img.cols, img.rows);
            
            if (armor_box.width > 0 && armor_box.height > 0) {
                boxes.push_back(armor_box);
                confidences.push_back(max_class_conf);
                class_ids.push_back(best_class_id);
            }
        }
    }

    std::vector<int> indices;
    if (!boxes.empty()) {
        cv::dnn::NMSBoxes(boxes, confidences, 0.6f, 0.45f, indices);
    }

    for (int idx : indices) {
        cv::Rect armor_box = boxes[idx];
        float conf = confidences[idx];
        int class_id = class_ids[idx];
        std::string class_name = (class_id >= 0 && class_id < class_names.size()) ? class_names[class_id] : "Unknown";

        // ================= 核心新增：PnP 解算 =================
        cv::Mat rvec, tvec;
        if (solver.solve(armor_box, rvec, tvec)) {
            double tx = tvec.at<double>(0, 0);
            double ty = tvec.at<double>(1, 0);
            double tz = tvec.at<double>(2, 0);

            std::cout << "  [YOLO] " << class_name << " | 坐标: (" 
                      << tx << ", " << ty << ", " << tz << ")" << std::endl;

            // 把解算坐标画在框下
            std::stringstream ss_pnp;
            ss_pnp << "t=(" << std::fixed << std::setprecision(2) << tz << "m)";
            cv::putText(img, ss_pnp.str(), cv::Point(armor_box.x, armor_box.y + armor_box.height + 15), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
        }
        // ======================================================

        cv::rectangle(img, armor_box, cv::Scalar(0, 255, 0), 2);
        std::stringstream ss;
        ss << class_name << " " << std::fixed << std::setprecision(2) << conf;
        cv::putText(img, ss.str(), cv::Point(armor_box.x, armor_box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }

    std::string save_path = "yolo_result_" + std::to_string(img_idx) + ".jpg";
    cv::imwrite(save_path, img);
    std::cout << "  结果已保存至 " << save_path << std::endl;
}

int main() {
    std::cout << "========== 启动纯血 YOLO 终极作业版 ==========" << std::endl;
    
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YoloTest");
    Ort::SessionOptions session_options;
    Ort::Session session(env, "../assets/yolo_armor.onnx", session_options);
    
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};

    std::vector<std::string> class_names = {"Sentinel", "Hero", "Engineer", "Infantry3", "Infantry4", "Infantry5", "Outpost", "BaseB", "BaseT"};

    // 实例化 PnP 解算器
    PnpSolver solver;

    // === 任务 1：图片测试与坐标解算 ===
    std::vector<std::string> image_names = {"test_img1.jpg", "test_img2.jpg", "test_img3.jpg"};
    for (size_t i = 0; i < image_names.size(); ++i) {
        std::string img_path = "../assets/" + image_names[i];
        processImage(img_path, session, input_names, output_names, class_names, solver, i + 1);
    }

    // === 任务 2：视频流处理 ===
    std::cout << "\n========== 开始处理视频 ==========" << std::endl;
    std::string video_path = "../assets/test_video.mp4"; 
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) return -1;

    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter writer("yolo_final_result.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frame_width, frame_height));

    cv::Mat img;
    int frame_count = 0;

    while (cap.read(img)) {
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(640, 480));
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        cv::Mat float_img;
        rgb.convertTo(float_img, CV_32F, 1.0 / 255.0); 

        std::vector<float> input_data(3 * 480 * 640);
        std::vector<cv::Mat> channels(3);
        cv::split(float_img, channels); 
        for (int c = 0; c < 3; ++c) {
            std::memcpy(input_data.data() + c * 480 * 640, channels[c].data, 480 * 640 * sizeof(float));
        }

        std::vector<int64_t> input_shape = {1, 3, 480, 640};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        float* floatarr = output_tensors[0].GetTensorMutableData<float>();

        int num_anchors = 6300;
        float scale_x = (float)img.cols / 640.0f;
        float scale_y = (float)img.rows / 480.0f;

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;

        for (int i = 0; i < num_anchors; ++i) {
            float max_class_conf = 0.0f;
            int best_class_id = -1;
            for (int c = 4; c <= 12; ++c) {
                float score = sigmoid(floatarr[c * num_anchors + i]);
                if (score > max_class_conf) {
                    max_class_conf = score;
                    best_class_id = c - 4;
                }
            }

            if (max_class_conf > 0.6f) {
                std::vector<cv::Point> pts;
                for (int k = 0; k < 4; ++k) {
                    float raw_x = floatarr[(13 + k * 2) * num_anchors + i];
                    float raw_y = floatarr[(14 + k * 2) * num_anchors + i];
                    int px = static_cast<int>(raw_x * scale_x);
                    int py = static_cast<int>(raw_y * scale_y);
                    pts.push_back(cv::Point(px, py));
                }

                cv::Rect armor_box = cv::boundingRect(pts);
                armor_box &= cv::Rect(0, 0, img.cols, img.rows);
                
                if (armor_box.width > 0 && armor_box.height > 0) {
                    boxes.push_back(armor_box);
                    confidences.push_back(max_class_conf);
                    class_ids.push_back(best_class_id);
                }
            }
        }

        std::vector<int> indices;
        if (!boxes.empty()) {
            cv::dnn::NMSBoxes(boxes, confidences, 0.6f, 0.45f, indices);
        }

        for (int idx : indices) {
            cv::Rect armor_box = boxes[idx];
            float conf = confidences[idx];
            int class_id = class_ids[idx];
            std::string class_name = (class_id >= 0 && class_id < class_names.size()) ? class_names[class_id] : "Unknown";

            // 视频流中也可以解算，但不打印以免刷屏
            cv::Mat rvec, tvec;
            solver.solve(armor_box, rvec, tvec);

            cv::rectangle(img, armor_box, cv::Scalar(0, 255, 0), 2);
            std::stringstream ss;
            ss << class_name << " " << std::fixed << std::setprecision(2) << conf;
            cv::putText(img, ss.str(), cv::Point(armor_box.x, armor_box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        }

        writer.write(img);

        if (frame_count % 60 == 0) {
            std::string debug_name = "yolo_frame_" + std::to_string(frame_count) + ".jpg";
            cv::imwrite(debug_name, img);
        }
        frame_count++;
    }

    cap.release();
    writer.release();
    std::cout << "视频处理完成！已保存为 yolo_final_result.mp4" << std::endl;
    return 0;
}