#include <rclcpp/rclcpp.hpp>
#include <string>
#include <std_msgs/msg/string.hpp>
#include <sstream>
#include <iostream>

#include "../include/rescue_vision_26/qr.hpp"

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<vision_rescue_26::QR>();
  node->run();
  rclcpp::shutdown();
  return 0;
}

namespace vision_rescue_26
{
  using namespace cv;
  using namespace std;

  QR::QR() : Node("qr"), isRecv(false)
  {
    init();
  }

  QR::~QR()
  {
  }

  bool QR::init()
  {
    // 파라미터 선언
    this->declare_parameter("camera_topic", "/camera/camera/color/image_raw");
    param = this->get_parameter("camera_topic").as_string();

    RCLCPP_INFO(this->get_logger(), "Starting Rescue Vision With Camera : %s", param.c_str());

    // Publisher와 Subscriber 설정
    img_result = this->create_publisher<sensor_msgs::msg::Image>("/qr", 10);
    img_sub = this->create_subscription<sensor_msgs::msg::Image>(param, 10, std::bind(&QR::imageCallBack, this, std::placeholders::_1));

    return true;
  }

  void QR::run()
  {
    rclcpp::Rate loop_rate(50);
    while (rclcpp::ok())
    {
      rclcpp::spin_some(shared_from_this());
      if (isRecv)
      {
        update();
        sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", output_qr).toImageMsg();
        img_result->publish(*msg);
      }
      loop_rate.sleep();
    }
  }

  void QR::imageCallBack(const sensor_msgs::msg::Image::SharedPtr msg_img)
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

  void QR::update()
  {
    clone_mat = original->clone();
    resize(clone_mat, clone_mat, Size(640, 480), 0, 0, INTER_CUBIC);
    cvtColor(clone_mat, gray_clone, COLOR_BGR2GRAY);

    output_qr = clone_mat.clone();

    if (detector.detect(gray_clone, points))
    {
      info = detector.decode(gray_clone, points);

      if (!info.empty())
      {
        cv::Size text_size = cv::getTextSize(info, cv::FONT_HERSHEY_SIMPLEX, 1, 2, nullptr);
        cv::Point2i bg_top_left(points[0].x, points[0].y - text_size.height - 20);
        cv::Point2i bg_bottom_right(bg_top_left.x + text_size.width, bg_top_left.y + text_size.height + 5);

        cv::rectangle(output_qr, bg_top_left, bg_bottom_right, cv::Scalar(0, 0, 0), -1);
        cv::putText(output_qr, info, cv::Point(points[0].x, points[0].y - 20), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        cv::polylines(output_qr, points, true, cv::Scalar(0, 0, 0), 5);
        //RCLCPP_INFO(this->get_logger(), "QR Code Info: %s", info.c_str());
      }
    }

    delete original;
    isRecv = false;
  }

} // namespace vision_rescue_26