#pragma once

// ------------------ 기본 ROS ------------------
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <custom_msgs/msg/bounding_boxes.hpp>
#include <custom_msgs/msg/sign_data.hpp>

// ------------------ OpenCV ------------------
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

// ------------------ STL ------------------
#include <mutex>
#include <vector>

// ------------------ TF2 ------------------
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <Eigen/Dense>
#include <tf2_eigen/tf2_eigen.hpp>

#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

// ------------------ 구조체 ------------------
struct Object
{
    std::string object_name;     // 객체 이름
    int id;                      // 객체 ID
    cv::Rect roi;                // 이미지 ROI

    cv::Point3f center_cam;      // camera 기준 좌표
    cv::Point3f center_base;     // base 기준 좌표

    int detect_count = 0;
    int lost_count = 0;

    bool confirmed = false;
    bool active = true;

    rclcpp::Time last_update;
};

// ------------------ 클래스 ------------------
class LocationCalculation : public rclcpp::Node
{
public:
    LocationCalculation();

private:
    // ------------------ 콜백 ------------------
    void colorCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void boxesCallback(const custom_msgs::msg::BoundingBoxes::SharedPtr msg);
    void boxesAprilCallback(const custom_msgs::msg::BoundingBoxes::SharedPtr msg);
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    void timerCallback();

    // ------------------ 함수 ------------------
    cv::Point3f getCenterXYZFromROI(const cv::Rect& roi);
    cv::Point3f transformToFrame(const cv::Point3f &pt_in, const std::string &target_frame, const std::string &source_frame);

    // ------------------ pub ------------------
    void publish_data(const Object &obj, const cv::Point3f &location);
    void draw_objects(const std::vector<Object> &objects);
    void publish_markers(const std::vector<Object> &objects);

    // ------------------ Subscriber ------------------
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<custom_msgs::msg::BoundingBoxes>::SharedPtr boxes_sub_;
    rclcpp::Subscription<custom_msgs::msg::BoundingBoxes>::SharedPtr april_boxes_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr caminfo_sub_;

    // ------------------ Publisher ------------------
    rclcpp::Publisher<custom_msgs::msg::SignData>::SharedPtr pub_location_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

    // ------------------ Timer ------------------
    rclcpp::TimerBase::SharedPtr timer_;

    // ------------------ TF ------------------
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::string target_frame_;
    std::string source_frame_;

    // ------------------ 데이터 ------------------
    cv::Mat color_img_;
    cv::Mat depth_img_;
    rclcpp::Time last_color_stamp_;

    std::vector<Object> objects_;
    std::vector<Object> objects_sign_;
    std::vector<Object> objects_april_;

    std::mutex data_mutex_;

    std::vector<cv::Scalar> base_colors_ = {
        cv::Scalar(0,0,255),
        cv::Scalar(0,255,0),
        cv::Scalar(255,0,0)
    };

    // ------------------ 카메라 파라미터 ------------------
    float fx_ = 0.0f;
    float fy_ = 0.0f;
    float cx_ = 0.0f;
    float cy_ = 0.0f;

    // ------------------ 옵션 ------------------
    bool show_window_ = true;
    bool pub_cloud_   = true;
    bool pub_marker_  = true;
};