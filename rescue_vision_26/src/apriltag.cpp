#include "../include/rescue_vision_26/apriltag.hpp"
#include <algorithm>

namespace apriltag
{

Apriltag::Apriltag() : Node("apriltag")
{
    // 파라미터 선언
    this->declare_parameter<std::string>("sub_topic", "/detections");
    this->declare_parameter<std::string>("pub_topic", "/april/bounding_boxes");

    this->get_parameter("sub_topic", sub_topic_);
    this->get_parameter("pub_topic", pub_topic_);

    // Subscriber
    sub_ = this->create_subscription<apriltag_msgs::msg::AprilTagDetectionArray>(
        sub_topic_, 10,
        std::bind(&Apriltag::detection_callback, this, std::placeholders::_1));

    // Publisher
    pub_ = this->create_publisher<custom_msgs::msg::BoundingBoxes>(
        pub_topic_, 10);

    RCLCPP_INFO(this->get_logger(),
        "Subscribed to: %s | Publishing to: %s",
        sub_topic_.c_str(), pub_topic_.c_str());
}

void Apriltag::detection_callback(
    const apriltag_msgs::msg::AprilTagDetectionArray::SharedPtr msg)
{
    custom_msgs::msg::BoundingBoxes out_msg;

    for (const auto &det : msg->detections)
    {
        custom_msgs::msg::BoundingBox box;

        // class_name
        box.class_name = det.family + "_" + std::to_string(det.id);

        // confidence
        box.confidence = 1.0;

        // bbox 계산
        double min_x = 1e9, max_x = -1e9;
        double min_y = 1e9, max_y = -1e9;

        for (const auto &p : det.corners)
        {
            min_x = std::min(min_x, p.x);
            max_x = std::max(max_x, p.x);
            min_y = std::min(min_y, p.y);
            max_y = std::max(max_y, p.y);
        }

        box.x1 = min_x;
        box.y1 = min_y;
        box.x2 = max_x;
        box.y2 = max_y;

        out_msg.boxes.push_back(box);
    }

    pub_->publish(out_msg);
}

} // namespace apriltag


int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<apriltag::Apriltag>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}