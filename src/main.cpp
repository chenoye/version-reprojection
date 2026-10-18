#include <iostream>
#include <opencv2/opencv.hpp>
#include "detector.hpp"
#include "solver.hpp"

int main() {
    std::cout << "========== 启动传统视觉最终交作业版 ==========" << std::endl;

    ArmorDetector detector("../assets/mlp.onnx");
    PnpSolver solver;

    // === 任务 1：批量测试三张图片 ===
    std::vector<std::string> image_names = {"test_img1.jpg", "test_img2.jpg", "test_img3.jpg"};
    for (size_t i = 0; i < image_names.size(); ++i) {
        std::string img_path = "../assets/" + image_names[i];
        cv::Mat img = cv::imread(img_path);
        if (img.empty()) continue;

        std::cout << "\n--- 正在处理: " << image_names[i] << " ---" << std::endl;

        std::vector<ArmorTarget> targets = detector.detect(img);

        for (auto& target : targets) {
            cv::Mat rvec, tvec;
            if (solver.solve(target.armor_box, rvec, tvec)) {
                target.tx = tvec.at<double>(0, 0);
                target.ty = tvec.at<double>(1, 0);
                target.tz = tvec.at<double>(2, 0);
                
                cv::rectangle(img, target.armor_box, cv::Scalar(0, 255, 0), 2);
                cv::putText(img, target.label, cv::Point(target.armor_box.x, target.armor_box.y - 10), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

                std::cout << "  装甲板: " << target.label << " 坐标: (" 
                          << target.tx << ", " << target.ty << ", " << target.tz << ")" << std::endl;
            }
        }

        std::string save_path = "traditional_result_" + std::to_string(i + 1) + ".jpg";
        cv::imwrite(save_path, img);
    }

    // === 任务 2：处理视频 ===
    std::cout << "\n========== 开始处理视频 ==========" << std::endl;
    std::string video_path = "../assets/test_video.mp4"; 
    cv::VideoCapture cap(video_path);
    if (cap.isOpened()) {
        int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fps = cap.get(cv::CAP_PROP_FPS);
        cv::VideoWriter writer("traditional_result_video.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frame_width, frame_height));

        cv::Mat frame;
        int frame_count = 0;
        while (cap.read(frame)) {
            std::vector<ArmorTarget> targets = detector.detect(frame);
            for (auto& target : targets) {
                cv::rectangle(frame, target.armor_box, cv::Scalar(0, 255, 0), 2);
                cv::putText(frame, target.label, cv::Point(target.armor_box.x, target.armor_box.y - 10), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            }
            writer.write(frame);

            if (frame_count % 60 == 0) {
                std::string debug_name = "trad_frame_" + std::to_string(frame_count) + ".jpg";
                cv::imwrite(debug_name, frame);
            }
            frame_count++;
        }
        cap.release();
        writer.release();
        std::cout << "视频处理完成！已保存为 traditional_result_video.mp4" << std::endl;
    }

    return 0;
}