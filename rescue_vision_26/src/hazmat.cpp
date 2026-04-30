#include <rclcpp/rclcpp.hpp>
#include <string>
#include <std_msgs/msg/string.hpp>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "../include/rescue_vision_26/hazmat.hpp"

constexpr float CONFIDENCE_THRESHOLD = 0.8;
constexpr float NMS_THRESHOLD = 0.4;
constexpr int NUM_CLASSES = 15;

const cv::Scalar colors[] = {{0, 255, 255}, {255, 255, 0}, {0, 255, 0}, {255, 0, 0}};
const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<vision_rescue_26::HAZMAT>();
    node->run();
    rclcpp::shutdown();
    return 0;
}

namespace vision_rescue_26
{
    using namespace cv;
    using namespace std;

    HAZMAT::HAZMAT() : Node("hazmat"), isRecv(false)
    {
        std::string packagePath = ament_index_cpp::get_package_share_directory("rescue_vision_26");
        RCLCPP_INFO(this->get_logger(), "Package path: %s", packagePath.c_str());
        std::string dir = packagePath + "/yolo/";

        {
            std::ifstream class_file(dir + "classes.txt");
            if (!class_file)
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to open classes.txt");
            }

            std::string line;
            while (std::getline(class_file, line))
                class_names.push_back(line);
        }

        std::string modelConfiguration = dir + "yolov7_tiny_hazmat.cfg";
        std::string modelWeights = dir + "2025_02_13.weights";

        net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        init();
    }

    HAZMAT::~HAZMAT()
    {
    }

    bool HAZMAT::init()
    {
        // 파라미터 선언
        this->declare_parameter("img_width", 640);
        this->declare_parameter("img_height", 360);

        img_width_ = this->get_parameter("img_width").as_int();
        img_height_ = this->get_parameter("img_height").as_int();

        this->declare_parameter("camera_topic", "/camera/camera/color/image_raw");
        param = this->get_parameter("camera_topic").as_string();

        RCLCPP_INFO(this->get_logger(), "Starting Rescue Vision With Camera : %s", param.c_str());

        // Publisher와 Subscriber 설정
        img_result = this->create_publisher<sensor_msgs::msg::Image>("/hazmat", 10);
        boxes_pub_ = this->create_publisher<custom_msgs::msg::BoundingBoxes>(
            "/sign/bounding_boxes", 10);
        img_sub = this->create_subscription<sensor_msgs::msg::Image>(
            param,
            rclcpp::SensorDataQoS(),
            std::bind(&HAZMAT::imageCallBack, this, std::placeholders::_1));

        return true;
    }

    void HAZMAT::run()
    {
        rclcpp::Rate loop_rate(50);

        while (rclcpp::ok())
        {
            rclcpp::spin_some(shared_from_this());
            loop_rate.sleep();

            if (isRecv)
            {
                update();

                custom_msgs::msg::BoundingBoxes msg_boxes;

                for (auto &d : detections)
                {
                    custom_msgs::msg::BoundingBox box;

                    box.class_name = d.name;
                    box.confidence = 1.0;

                    box.x1 = d.roi.x;
                    box.y1 = d.roi.y;
                    box.x2 = d.roi.x + d.roi.width;
                    box.y2 = d.roi.y + d.roi.height;

                    msg_boxes.boxes.push_back(box);
                }

                boxes_pub_->publish(msg_boxes);

                sensor_msgs::msg::Image::SharedPtr msg =
                    cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
                img_result->publish(*msg);
            }
        }
    }

    void HAZMAT::imageCallBack(const sensor_msgs::msg::Image::SharedPtr msg_img)
    {
        if (!isRecv)
        {
            original = new cv::Mat(cv_bridge::toCvCopy(msg_img, "bgr8")->image);
            if (original != NULL)
            {
                isRecv = true;
            }
        }
    }

    void HAZMAT::update()
    {
        clone_mat = original->clone();
        cv::resize(clone_mat, clone_mat, cv::Size(img_width_, img_height_), 0, 0, cv::INTER_CUBIC);
        set_yolo();
        delete original;
        isRecv = false;
    }

    void HAZMAT::set_yolo()
    {
        detections.clear();

        std::vector<cv::Mat> yolo_outputs;
        auto output_names = net.getUnconnectedOutLayersNames();
        frame = clone_mat.clone();

        auto total_start = std::chrono::steady_clock::now();
        cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(416, 416), cv::Scalar(), true, false, CV_32F);
        net.setInput(blob);

        auto dnn_start = std::chrono::steady_clock::now();
        net.forward(yolo_outputs, output_names);
        auto dnn_end = std::chrono::steady_clock::now();

        std::vector<int> indices[NUM_CLASSES];
        std::vector<cv::Rect> boxes[NUM_CLASSES];
        std::vector<float> scores[NUM_CLASSES];

        for (auto &output : yolo_outputs)
        {
            const auto num_boxes = output.rows;
            for (int i = 0; i < num_boxes; i++)
            {
                auto x = output.at<float>(i, 0) * frame.cols; // 중심 x
                auto y = output.at<float>(i, 1) * frame.rows; // 중심 y
                auto width = output.at<float>(i, 2) * frame.cols;
                auto height = output.at<float>(i, 3) * frame.rows;
                cv::Rect rect(x - width / 2, y - height / 2, width, height);

                for (int c = 0; c < NUM_CLASSES; c++)
                {
                    auto confidence = *output.ptr<float>(i, 5 + c);
                    if (confidence >= CONFIDENCE_THRESHOLD)
                    {
                        boxes[c].push_back(rect);
                        scores[c].push_back(confidence);
                    }
                }
            }
        }

        for (int c = 0; c < NUM_CLASSES; c++)
            cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);

        for (int c = 0; c < NUM_CLASSES; c++)
        {
            for (size_t i = 0; i < indices[c].size(); ++i)
            {
                const auto color = colors[c % NUM_COLORS];
                auto idx = indices[c][i];
                const auto &rect = boxes[c][idx];

                // Check for overlapping boxes of the same class
                isOverlapping = false;
                if (indices[c].size() != 0)
                {
                    for (size_t j = 0; j < indices[c].size(); ++j)
                    {
                        if (j != i)
                        {
                            auto idx2 = indices[c][j];
                            const auto &rect2 = boxes[c][idx2];
                            if (isRectOverlapping(rect, rect2))
                            {
                                if (rect2.area() < rect.area())
                                {
                                    isOverlapping = true;
                                    break;
                                }
                                else
                                {
                                    continue;
                                }
                            }
                        }
                    }
                }

                if (!isOverlapping)
                {
                    Detection det;
                    det.name = class_names[c];
                    det.roi = rect;
                    detections.push_back(det);

                    cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);

                    std::ostringstream label_ss;
                    label_ss << class_names[c] << ": " << std::fixed << std::setprecision(2) << scores[c][idx];
                    auto label = label_ss.str();

                    int baseline;
                    auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
                    cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10),
                                  cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
                    cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
                                cv::Scalar(0, 0, 0));
                }
            }
        }

        auto total_end = std::chrono::steady_clock::now();

        float inference_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(dnn_end - dnn_start).count();
        float total_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
        std::ostringstream stats_ss;
        stats_ss << std::fixed << std::setprecision(2);
        stats_ss << "Inference FPS: " << inference_fps << ", Total FPS: " << total_fps;
        auto stats = stats_ss.str();
    }

    bool HAZMAT::isRectOverlapping(const cv::Rect &rect1, const cv::Rect &rect2)
    {
        int x1 = std::max(rect1.x, rect2.x);
        int y1 = std::max(rect1.y, rect2.y);
        int x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
        int y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);

        if (x1 < x2 && y1 < y2)
            return true;
        else
            return false;
    }

} // namespace vision_rescue_26
