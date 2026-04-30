#pragma once

#include <rclcpp/rclcpp.hpp>
#include <apriltag_msgs/msg/april_tag_detection_array.hpp>
#include <custom_msgs/msg/bounding_boxes.hpp>

namespace apriltag
{

class Apriltag : public rclcpp::Node
{
public:
    Apriltag();

private:
    void detection_callback(
        const apriltag_msgs::msg::AprilTagDetectionArray::SharedPtr msg);

    // sub / pub
    rclcpp::Subscription<apriltag_msgs::msg::AprilTagDetectionArray>::SharedPtr sub_;
    rclcpp::Publisher<custom_msgs::msg::BoundingBoxes>::SharedPtr pub_;

    // params
    std::string sub_topic_;
    std::string pub_topic_;
};

}