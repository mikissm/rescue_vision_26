#include "rclcpp/rclcpp.hpp"
#include <string>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>

#include <iostream>
#include <vector>
#include "std_msgs/msg/string.hpp"
#include <opencv2/opencv.hpp>

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include "custom_msgs/msg/bounding_boxes.hpp"
#include "custom_msgs/msg/bounding_box.hpp"

using namespace cv;
using namespace std;

namespace vision_rescue_26
{

struct Detection
{
    std::string name;  // 객체 이름 (YOLO class)
    cv::Rect roi;      // bounding box
};

class HAZMAT : public rclcpp::Node
{
public:
  HAZMAT();  // ROS2에서는 argc, argv를 받지 않습니다
  ~HAZMAT();

  bool init();
  void run();
  void update();
  bool isRecv;

  Mat* original;
  Mat clone_mat;
  Mat frame;  //output_hazmat
  Mat blob;

  int img_width_;
  int img_height_;

  cv::dnn::Net net;
  std::vector<std::string> class_names;
  bool isOverlapping;  // 겹침 여부 플래그
  bool isRectOverlapping(const cv::Rect& rect1, const cv::Rect& rect2);
  void set_yolo();

  std::string param;
  String info;

private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_result;
  rclcpp::Publisher<custom_msgs::msg::BoundingBoxes>::SharedPtr boxes_pub_;
  void imageCallBack(const sensor_msgs::msg::Image::SharedPtr msg_img);

  std::vector<Detection> detections;
};

}  // namespace vision_rescue_26