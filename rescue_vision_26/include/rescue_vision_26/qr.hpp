#include "rclcpp/rclcpp.hpp"
#include <string>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>

#include <iostream>
#include <vector>
#include "std_msgs/msg/string.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

namespace vision_rescue_26
{
class QR : public rclcpp::Node
{
public:
  QR();
  ~QR();

  bool init();
  void run();
  void update();
  bool isRecv;

  Mat* original;
  Mat clone_mat;
  Mat gray_clone;
  Mat output_qr;

  vector<Point> points;
  QRCodeDetector detector;

  std::string param;
  String info;

private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_result;
  void imageCallBack(const sensor_msgs::msg::Image::SharedPtr msg_img);
};

}  // namespace vision_rescue_26