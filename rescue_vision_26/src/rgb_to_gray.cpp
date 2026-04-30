#include "../include/rescue_vision_26/rgb_to_gray.hpp"

namespace vision
{

GrayscaleConverter::GrayscaleConverter()
: Node("rgb_to_gray")
{
    this->declare_parameter<std::string>("sub_topic", "/camera/camera/color/image_raw");
    this->declare_parameter<std::string>("pub_topic", "/image_mono");

    this->get_parameter("sub_topic", sub_topic_);
    this->get_parameter("pub_topic", pub_topic_);

    sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        sub_topic_, 10,
        std::bind(&GrayscaleConverter::image_callback, this, std::placeholders::_1));

    pub_ = this->create_publisher<sensor_msgs::msg::Image>(
        pub_topic_, 10);

    RCLCPP_INFO(this->get_logger(),
        "Grayscale node started | %s -> %s",
        sub_topic_.c_str(), pub_topic_.c_str());
}

void GrayscaleConverter::image_callback(
    const sensor_msgs::msg::Image::SharedPtr msg)
{
    try
    {
        cv_bridge::CvImagePtr cv_ptr =
            cv_bridge::toCvCopy(msg, "bgr8");

        cv::Mat gray;
        cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGR2GRAY);

        sensor_msgs::msg::Image out_msg =
            *cv_bridge::CvImage(
                msg->header,
                "mono8",
                gray
            ).toImageMsg();

        pub_->publish(out_msg);
    }
    catch (const cv_bridge::Exception &e)
    {
        RCLCPP_ERROR(this->get_logger(),
            "cv_bridge error: %s", e.what());
    }
}

} // namespace vision

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<vision::GrayscaleConverter>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}