#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "armor_tracker.hpp"
#include "solver.hpp"

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

int main() {
    std::cout << "========== 启动 EKF 融合自瞄大作业 ==========" << std::endl;
    
    // 1. 初始化 ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YoloTest");
    Ort::SessionOptions session_options;
    Ort::Session session(env, "../assets/yolo_armor.onnx", session_options);
    
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};

    // 2. 初始化 PnP 与 EKF 模块
    PnpSolver solver;
    ArmorTracker tracker;

    // 3. 加载 IMU 数据
    std::string imu_path = "../assets/2026-05-20_23-57-31.txt";
    std::vector<ImuData> imu_data = tracker.loadImuData(imu_path);
    if (imu_data.empty()) return -1;

    // 4. 加载视频
    std::string video_path = "../assets/2026-05-20_23-57-31.avi"; 
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) return -1;

    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    cv::VideoWriter writer("ekf_result.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frame_width, frame_height));

    cv::Mat img;
    int frame_count = 0;
    double video_start_time = imu_data.front().timestamp;

    // 相机内参和畸变系数
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 
        2362.2795952460974, 0.0, 679.17513921636282, 
        0.0, 2336.1916035376717, 529.06858905837635, 
        0.0, 0.0, 1.0);
    cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 
        -0.098561592108101559, 0.11173066531932681, 
        -0.0038352945867108721, -0.0020421443354436304, 0.0);

    while (cap.read(img)) {
        // ================= A. 时间同步 =================
        double current_video_time = video_start_time + (frame_count / fps);
        double target_imu_time = current_video_time - 0.009; // 9ms 延迟补偿

        ImuData current_imu = imu_data.back(); 
        for (const auto& imu : imu_data) {
            if (imu.timestamp >= target_imu_time) {
                current_imu = imu;
                break;
            }
        }

        double w = current_imu.w, x = current_imu.x, y = current_imu.y, z = current_imu.z;
        double siny_cosp = 2 * (w * z + x * y);
        double cosy_cosp = 1 - 2 * (y * y + z * z);
        float imu_yaw = std::atan2(siny_cosp, cosy_cosp);

        // ================= B. YOLO 推理与极角排序 =================
        cv::Mat resized, rgb, float_img;
        cv::resize(img, resized, cv::Size(640, 480));
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
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
        
        std::vector<std::vector<cv::Point2f>> all_armors_pts;
        std::vector<float> confidences;

        for (int i = 0; i < num_anchors; ++i) {
            float max_class_conf = 0.0f;
            for (int c = 4; c <= 12; ++c) {
                float score = sigmoid(floatarr[c * num_anchors + i]);
                if (score > max_class_conf) max_class_conf = score;
            }
            if (max_class_conf > 0.6f) {
                std::vector<cv::Point2f> pts;
                for (int k = 0; k < 4; ++k) {
                    float raw_x = floatarr[(13 + k * 2) * num_anchors + i];
                    float raw_y = floatarr[(14 + k * 2) * num_anchors + i];
                    pts.push_back(cv::Point2f(raw_x * scale_x, raw_y * scale_y));
                }
                
                // 极角排序：解决 PnP 解算中角点顺序错乱导致的姿态翻转
                cv::Point2f center(0, 0);
                for (const auto& p : pts) { center += p; }
                center.x /= 4.0f; center.y /= 4.0f;
                std::sort(pts.begin(), pts.end(), [center](const cv::Point2f& a, const cv::Point2f& b) {
                    return std::atan2(a.y - center.y, a.x - center.x) < std::atan2(b.y - center.y, b.x - center.x);
                });

                all_armors_pts.push_back(pts);
                confidences.push_back(max_class_conf);
            }
        }
        
        std::vector<int> indices;
        if (!all_armors_pts.empty()) {
            std::vector<cv::Rect> boxes;
            for (const auto& pts : all_armors_pts) boxes.push_back(cv::boundingRect(pts));
            cv::dnn::NMSBoxes(boxes, confidences, 0.6f, 0.45f, indices);
        }

        // ================= C. 送入 EKF 并保存当前位姿 =================
        cv::Mat latest_state;
        bool has_target = false;
        cv::Mat best_rvec, best_tvec;
        std::vector<cv::Rect> blue_boxes; 
        
        for (int idx : indices) {
            std::vector<cv::Point2f> armor_pts = all_armors_pts[idx];
            cv::Rect draw_box = cv::boundingRect(armor_pts);
            blue_boxes.push_back(draw_box);
            
            cv::Mat rvec, tvec;
            if (solver.solve(armor_pts, rvec, tvec)) {
                cv::Mat R;
                cv::Rodrigues(rvec, R); 
                float board_yaw = std::atan2(R.at<double>(0, 2), R.at<double>(2, 2));
                
                latest_state = tracker.predictAndUpdate(tvec, board_yaw, imu_yaw);
                tracker.saveDataToCsv(latest_state);

                // 绘制真实观测蓝框
                cv::rectangle(img, draw_box, cv::Scalar(255, 0, 0), 2);
                
                if (!has_target) {
                    best_rvec = rvec.clone();
                    best_tvec = tvec.clone();
                    has_target = true;
                }
            }
        }

        // ================= C.5 虚拟重投影与 2D 像素强吸附 =================
        if (has_target && !latest_state.empty()) {
            float radius = 0.2f; 
            
            cv::Mat R_0;
            cv::Rodrigues(best_rvec, R_0);
            cv::Mat N_0 = R_0.col(2);
            N_0 /= cv::norm(N_0);

            double nx = N_0.at<double>(0, 0);
            double nz = N_0.at<double>(2, 0);
            double n_norm = std::sqrt(nx * nx + nz * nz);
            
            if (n_norm > 1e-5) {
                double norm_x = nx / n_norm;
                double norm_z = nz / n_norm;
                double theta0 = std::atan2(norm_x, norm_z);

                // 提取 3D 锚点（当前蓝框高度与平滑后深度）
                double car_y = best_tvec.at<double>(1, 0); 
                double car_z = latest_state.at<float>(2); 
                double car_x = best_tvec.at<double>(0, 0) - radius * norm_x;

                cv::Point2f anchor_center_2d(0, 0);
                if (!blue_boxes.empty()) {
                    anchor_center_2d.x = blue_boxes[0].x + blue_boxes[0].width / 2.0f;
                    anchor_center_2d.y = blue_boxes[0].y + blue_boxes[0].height / 2.0f;
                }

                std::vector<cv::Point3f> pts_3d_local = {
                    cv::Point3f(-0.0675f, -0.0275f, 0),
                    cv::Point3f(-0.0675f,  0.0275f, 0),
                    cv::Point3f( 0.0675f,  0.0275f, 0),
                    cv::Point3f( 0.0675f, -0.0275f, 0)
                };

                for (int i = 0; i < 4; ++i) {
                    double angle = theta0 + i * CV_PI / 2.0;
                    
                    double Ni_x = std::sin(angle);
                    double Ni_z = std::cos(angle);

                    double bx = car_x + radius * Ni_x;
                    double bz = car_z + radius * Ni_z;
                    double by = car_y; 

                    if (bz < 0.2) continue;

                    cv::Mat Y_axis = (cv::Mat_<double>(3, 1) << 0, 1, 0);
                    cv::Mat Z_axis = (cv::Mat_<double>(3, 1) << Ni_x, 0, Ni_z);
                    cv::Mat X_axis = Y_axis.cross(Z_axis);

                    cv::Mat R_i(3, 3, CV_64F);
                    X_axis.copyTo(R_i.col(0));
                    Y_axis.copyTo(R_i.col(1));
                    Z_axis.copyTo(R_i.col(2));

                    std::vector<cv::Point2f> proj_pts;
                    cv::Mat rvec_i;
                    cv::Rodrigues(R_i, rvec_i); 
                    cv::Mat tvec_i = (cv::Mat_<double>(3, 1) << bx, by, bz);
                    
                    cv::projectPoints(pts_3d_local, rvec_i, tvec_i, cameraMatrix, distCoeffs, proj_pts);

                    // 2D 像素强吸附：垂直方向贴合
                    cv::Point2f proj_center(0, 0);
                    for (const auto& pt : proj_pts) proj_center += pt;
                    proj_center.x /= 4.0f; proj_center.y /= 4.0f;

                    float dy = anchor_center_2d.y - proj_center.y;
                    for (int k = 0; k < 4; ++k) {
                        proj_pts[k].y += dy;
                    }

                    // 越界过滤：防透视爆炸
                    cv::Rect rect_i = cv::boundingRect(proj_pts);
                    if (rect_i.width > img.cols * 0.6f || rect_i.height > img.rows * 0.6f) continue;
                    if (rect_i.x < -100 || rect_i.y < -100 || rect_i.x > img.cols || rect_i.y > img.rows) continue;

                    // 蓝红交接：重合即隐藏虚拟框
                    cv::Point2f final_center(0, 0);
                    for (const auto& pt : proj_pts) final_center += pt;
                    final_center.x /= 4.0f; final_center.y /= 4.0f;

                    bool overlap = false;
                    for (const auto& bbox : blue_boxes) {
                        cv::Point2f b_center(bbox.x + bbox.width/2.0f, bbox.y + bbox.height/2.0f);
                        if (cv::norm(final_center - b_center) < 80.0f) {
                            overlap = true;
                            break;
                        }
                    }

                    if (!overlap) {
                        for (int k = 0; k < 4; ++k) {
                            cv::line(img, proj_pts[k], proj_pts[(k+1)%4], cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                        }
                    }
                }
            }
        }

        // ================= D. 右上角状态显示 =================
        if (!latest_state.empty()) {
            float ex = latest_state.at<float>(0);
            float ey = latest_state.at<float>(1);
            float ez = latest_state.at<float>(2);
            float eyaw = latest_state.at<float>(3);

            std::stringstream ui_ss;
            ui_ss << "EKF -> X:" << std::fixed << std::setprecision(2) << ex 
                  << "  Y:" << ey 
                  << "  Z:" << ez 
                  << "  Yaw:" << eyaw;
            
            std::string ui_text = ui_ss.str();
            int fontFace = cv::FONT_HERSHEY_SIMPLEX;
            double fontScale = 0.7;
            int thickness = 2;
            
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(ui_text, fontFace, fontScale, thickness, &baseline);
            int text_x = img.cols - text_size.width - 20;
            if (text_x < 10) text_x = 10;
            
            cv::rectangle(img, cv::Rect(text_x - 5, 20, text_size.width + 10, text_size.height + 15), 
                          cv::Scalar(0, 0, 0), cv::FILLED);
            
            cv::putText(img, ui_text, cv::Point(text_x, 20 + text_size.height + 5), 
                        fontFace, fontScale, cv::Scalar(0, 255, 255), thickness);
        }

        writer.write(img);
        frame_count++;
        
        // ================= E. 终端动态进度条 =================
        if (frame_count % 10 == 0 || frame_count == total_frames) {
            double progress = (double)frame_count / total_frames * 100.0;
            std::cout << "\r[进度 " << std::fixed << std::setprecision(1) << progress << "%] "
                      << "Frame: " << frame_count << "/" << total_frames 
                      << " | 目标数: " << indices.size();
            
            if (!latest_state.empty()) {
                std::cout << " | EKF Z: " << std::setprecision(2) << latest_state.at<float>(2);
            }
            std::cout << std::flush;
        }
    }
    
    std::cout << "\n========== 处理完成！生成 ekf_result.mp4 ==========" << std::endl;

    cap.release();
    writer.release();
    return 0;
}