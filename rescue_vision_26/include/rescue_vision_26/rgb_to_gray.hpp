#ifndef RESCUE_VISION_26_GRAYSCALE_CONVERTER_HPP_
#define RESCUE_VISION_26_GRAYSCALE_CONVERTER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <string>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

namespace vision
{

class GrayscaleConverter : public rclcpp::Node
{
public:
    GrayscaleConverter();

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;

    std::string sub_topic_;
    std::string pub_topic_;
};

} // namespace vision

#endif