#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

int main() {
    std::cout << "========== 纯血 YOLO 测试 (丢弃 MLP) ==========" << std::endl;
    
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YoloTest");
    Ort::SessionOptions session_options;
    Ort::Session session(env, "../assets/yolo_armor.onnx", session_options);
    
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};

    // 深大官方定义的 9 个类别
    std::vector<std::string> class_names = {"Sentinel", "Hero", "Engineer", "Infantry3", "Infantry4", "Infantry5", "Outpost", "BaseB", "BaseT"};

    std::string video_path = "../assets/test_video.mp4"; 
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) return -1;

    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter writer("yolo_result_video.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frame_width, frame_height));

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
            memory_info, input_data.data(), input_data.size(), 
            input_shape.data(), input_shape.size());

        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        float* floatarr = output_tensors[0].GetTensorMutableData<float>();

        int num_anchors = 6300;
        float scale_x = (float)img.cols / 640.0f;
        float scale_y = (float)img.rows / 480.0f;

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;

        for (int i = 0; i < num_anchors; ++i) {
            // 【核心改变】直接读取 ch4~12 的类别得分
            float max_class_conf = 0.0f;
            int best_class_id = -1;
            for (int c = 4; c <= 12; ++c) {
                float score = sigmoid(floatarr[c * num_anchors + i]);
                if (score > max_class_conf) {
                    max_class_conf = score;
                    best_class_id = c - 4; // 映射到 0~8
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

            // 画大框
            cv::rectangle(img, armor_box, cv::Scalar(0, 255, 0), 2);

            // 打标签（直接输出 YOLO 识别的车体类型）
            std::stringstream ss;
            ss << class_name << " " << std::fixed << std::setprecision(2) << conf;
            cv::putText(img, ss.str(), cv::Point(armor_box.x, armor_box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            
            if (frame_count % 30 == 0) {
                std::cout << "YOLO检测到: " << class_name << " 置信度: " << conf << std::endl;
            }
        }

        writer.write(img);

        if (frame_count % 60 == 0) {
            std::string debug_name = "yolo_debug_frame_" + std::to_string(frame_count) + ".jpg";
            cv::imwrite(debug_name, img);
        }
        frame_count++;
    }

    cap.release();
    writer.release();
    return 0;
}