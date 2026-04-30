#include <rclcpp/rclcpp.hpp>
#include <string>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/int32.hpp>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <map>

#include "../include/rescue_vision_26/master.hpp"

constexpr float CONFIDENCE_THRESHOLD = 0.5;
constexpr float NMS_THRESHOLD = 0.4;
constexpr int NUM_CLASSES = 15;
const cv::Scalar colors[] = {{0, 255, 255}, {255, 255, 0}, {0, 255, 0}, {255, 0, 0}};
const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);

#define PI 3.141592

float distance(float x1, float y1, float x2, float y2) {
    return std::sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}

inline bool safe_rect(const cv::Mat& m, cv::Rect& r)
{
  if (r.x < 0 || r.y < 0 ||
      r.x + r.width > m.cols ||
      r.y + r.height > m.rows ||
      r.width <= 0 || r.height <= 0)
    return false;
  return true;
}


int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<vision_rescue_26::MASTER>();
  node->run();
  rclcpp::shutdown();
  return 0;
}

namespace vision_rescue_26
{
  using namespace cv;
  using namespace std;

  MASTER::MASTER() : Node("master"), isRecv(false), isRecv_thermal(false), c_direction_found(false), rotation_enabled(false), is_first(true)
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

  MASTER::~MASTER()
  {
  }

  bool MASTER::init()
  {
    // 파라미터 선언
    this->declare_parameter("camera_topic", "/camera/camera/color/image_raw");
    this->declare_parameter("thermal_camera_topic", "/thermal_camera/image_colored");

    auto camera_topic = this->get_parameter("camera_topic").as_string();
    auto thermal_topic = this->get_parameter("thermal_camera_topic").as_string();

    RCLCPP_INFO(this->get_logger(), "Camera topics: %s, %s", camera_topic.c_str(), thermal_topic.c_str());

    // Publisher와 Subscriber 설정
    img_result = this->create_publisher<sensor_msgs::msg::Image>("/victim", 10);
    img_result_thermal = this->create_publisher<sensor_msgs::msg::Image>("/img_result_thermal", 10);

    img_sub = this->create_subscription<sensor_msgs::msg::Image>(camera_topic, 30, std::bind(&MASTER::imageCallBack, this, std::placeholders::_1));

    subscription_ = this->create_subscription<custom_msgs::msg::BoundingBoxes>("/yolo/bounding_boxes", 10, std::bind(&MASTER::bbox_callback, this, std::placeholders::_1));

    img_sub_thermal = this->create_subscription<sensor_msgs::msg::Image>(thermal_topic, 10, std::bind(&MASTER::imageCallBack_thermal, this, std::placeholders::_1));

    circle_c_.resize(2);
    circle_square_.resize(2);

    return true;
  }

  void MASTER::run()
  {
    rclcpp::Rate loop_rate(30);
    while (rclcpp::ok())
    {
      rclcpp::spin_some(shared_from_this());
      if (isRecv)
      {
        // if (isRecv_thermal)
        // {
        //   set_thermal();
        //   sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", thermal_mat).toImageMsg();
        //   img_result_thermal->publish(*msg);
        // }
        update();
      }

      if (isRecv_thermal)
      {
        set_thermal();
        sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", thermal_mat).toImageMsg();
        img_result_thermal->publish(*msg);
      }
      loop_rate.sleep();
    }
  }

  void MASTER::imageCallBack(const sensor_msgs::msg::Image::SharedPtr msg_img)
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

  void MASTER::imageCallBack_thermal(const sensor_msgs::msg::Image::SharedPtr msg_img)
  {
    if (!isRecv_thermal)
    {
      original_thermal = new cv::Mat(cv_bridge::toCvCopy(msg_img, "bgr8")->image);
      if (original_thermal != NULL)
      {
        isRecv_thermal = true;
      }
    }
  }

  void MASTER::bbox_callback(const custom_msgs::msg::BoundingBoxes::SharedPtr msg)
  {
    std::vector<BBoxData> circle_new;
    std::vector<BBoxData> square_new;

    // 함수 내부 static 상태 (히스토리 유지용)
    static std::deque<cv::Point2f> hist;
    static cv::Point2f prev_center(-1, -1);
    const int MAX_H = 3;

    float scale_x = (float)clone_mat.cols / 1280.0f;
    float scale_y = (float)clone_mat.rows / 720.0f;
    square_color = 0;

    // -------------------- YOLO 입력 파싱 --------------------
    for (const auto &box : msg->boxes)
    {
      BBoxData data;
      data.class_name = box.class_name;
      data.confidence = box.confidence;
      data.x1 = box.x1 * scale_x;
      data.y1 = box.y1 * scale_y;
      data.x2 = box.x2 * scale_x;
      data.y2 = box.y2 * scale_y;
      data.cx = (data.x1 + data.x2) / 2.0f;
      data.cy = (data.y1 + data.y2) / 2.0f;
      data.valid = true;

      if (box.class_name == "circle_c")
        circle_new.push_back(data);
      else if (box.class_name == "circle_square")
      {
        square_new.push_back(data);
        square_color = 1;
      }
      else if (box.class_name == "circle_square_black")
      {
        square_new.push_back(data);
        square_color = 2;
      }
    }

    // -------------------- 사각형 박스 관리 --------------------
    square_boxes.clear();
    updateBoxes(circle_square_, square_new);
    int count = 0;
    for (const auto &slot : circle_square_)
    {
      if (!slot.valid) continue;
      square_boxes.push_back(slot);
      if (++count >= 2) break;
    }

    // -------------------- Circle 없음 → 전체 상태 초기화 --------------------
    if (circle_new.empty())
    {
      // RCLCPP_WARN(this->get_logger(), "No circle detected — reset all states");

      circle_c_.clear();
      circles.clear();
      find_two = false;

      best_center = cv::Point(-1, -1);
      best_radius = 0;
      best_stable_count = 0;
      c_locked = false;

      c_rotation_started = false;
      c_stable_count = 0;
      first_c_latched = false;

      angleFrequency1.clear();
        // RCLCPP_WARN(this->get_logger(),
        //   "Different C detected (dist=%.1f). Resetting state.", dist);

        // --- 객체 변경 시 상태 초기화 ---
        hist.clear();
        circle_c_.clear();
        circles.clear();
        find_two = false;

        best_center = cv::Point(-1, -1);
        best_radius = 0;
      angleFrequency2.clear();
      angleFrequency3.clear();
      movement_count = 0;
      c_dir_flag = false;
      c_direction_found = false;
      rotation_enabled = false;
      result_maxAngle1 = result_maxAngle2 = result_maxAngle3 = 0;
      c = cv::Vec3f(0, 0, 0);

      hist.clear();
      prev_center = cv::Point2f(-1, -1);

      return;
    }

    // -------------------- 회전 감지 --------------------
    bool detect_rotation = square_new.empty();

    cv::Point2f cur(circle_new[0].cx, circle_new[0].cy);

    // 🔸 다른 물체 들어왔는지 검사 (중심 거리 기준)
    if (prev_center.x != -1)
    {
      float dist = cv::norm(cur - prev_center);
      if (dist > 50.0f)  // 너무 멀면 다른 객체로 판단
      {
        // RCLCPP_WARN(this->get_logger(),
        //   "Different C detected (dist=%.1f). Resetting state.", dist);

        // --- 객체 변경 시 상태 초기화 ---
        hist.clear();
        circle_c_.clear();
        circles.clear();
        find_two = false;

        best_center = cv::Point(-1, -1);
        best_radius = 0;
        best_stable_count = 0;
        c_locked = false;

        c_rotation_started = false;
        c_stable_count = 0;
        first_c_latched = false;

        angleFrequency1.clear();
        angleFrequency2.clear();
        angleFrequency3.clear();
        movement_count = 0;
        c_dir_flag = false;
        c_direction_found = false;
        rotation_enabled = false;
        result_maxAngle1 = result_maxAngle2 = result_maxAngle3 = 0;
        c = cv::Vec3f(0, 0, 0);
      }
    }
    prev_center = cur;

    hist.push_back(cur);
    if (hist.size() > MAX_H) hist.pop_front();

    // 기본값: 정지 상태
    c_rotation_state = 0;

    if (detect_rotation && hist.size() == 3)
    {
      cv::Point2f A = hist[0], B = hist[1], C = hist[2];
      float cross = (B.x - A.x) * (C.y - B.y) - (B.y - A.y) * (C.x - B.x);
      float dist = cv::norm(B - C);

      if (dist > 3.0f)
      {
        if (cross < 0)
        {
          c_rotation_state = 1; // CW
          //RCLCPP_INFO(this->get_logger(), "C rotating CW (no squares)");
        }
        else if (cross > 0)
        {
          c_rotation_state = 2; // CCW
          //RCLCPP_INFO(this->get_logger(), "C rotating CCW (no squares)");
        }
      }
    }
    else if (!detect_rotation)
    {
      c_rotation_state = 0;
    }

    // -------------------- 회전 중이면 C 갱신 X --------------------
    if (c_rotation_state != 0)
      return;

    // -------------------- 회전 안 하면 → 작은 원 1개만 circle_c_에 저장 --------------------
    circle_c_.clear();
    if (circle_new.size() == 1)
    {
      circle_c_.push_back(circle_new[0]);
    }
    else
    {
      float area1 = (circle_new[0].x2 - circle_new[0].x1) * (circle_new[0].y2 - circle_new[0].y1);
      float area2 = (circle_new[1].x2 - circle_new[1].x1) * (circle_new[1].y2 - circle_new[1].y1);
      circle_c_.push_back(area1 < area2 ? circle_new[0] : circle_new[1]);
    }

    // -------------------- 안정된 C 저장 --------------------
    best_circle = circle_c_[0];
    // RCLCPP_INFO(this->get_logger(),
    //             "C stable saved: center(%.1f, %.1f), rotation=0",
    //             best_circle.cx, best_circle.cy);
  }

  void MASTER::updateBoxes(std::vector<BBoxData> &slots, const std::vector<BBoxData> &new_boxes)
  {
    const float DIST_THRESHOLD = 50.0f;  // 가까운 기준 거리 (픽셀 단위)

    // 새 프레임에서 모든 슬롯 일단 비활성화, 나중에 새 박스와 매칭하면 활성화
    std::vector<bool> updated(slots.size(), false);

    for (const auto &nb : new_boxes)
    {
      // 기존 슬롯 중 가장 가까운 것 찾기
      float min_dist = 1e9;
      int best_idx = -1;

      for (size_t i = 0; i < slots.size(); ++i)
      {
        if (updated[i]) continue;  // 이미 새 박스로 갱신된 슬롯은 건너뜀
        float d = distance(slots[i].cx, slots[i].cy, nb.cx, nb.cy);
        if (d < min_dist)
        {
          min_dist = d;
          best_idx = static_cast<int>(i);
        }
      }

      if (best_idx != -1 && min_dist < DIST_THRESHOLD)
      {
        // 기존 슬롯 갱신
        slots[best_idx] = nb;
        updated[best_idx] = true;
      }
      else
      {
        // 새로운 객체 → 비어있는 슬롯에 넣기
        bool inserted = false;
        for (size_t i = 0; i < slots.size(); ++i)
        {
          if (!updated[i]) // 아직 갱신 안 된 슬롯
          {
            slots[i] = nb;
            updated[i] = true;
            inserted = true;
            break;
          }
        }
        // 슬롯이 모자라면 slots에 새로 추가
        if (!inserted)
        {
          slots.push_back(nb);
          updated.push_back(true);
        }
      }
    }

    // 업데이트 안 된 슬롯은 valid = false
    for (size_t i = 0; i < slots.size(); ++i)
    {
      slots[i].valid = updated[i];
    }

    // 오래된(유효하지 않은) 박스 제거
    slots.erase(
      std::remove_if(slots.begin(), slots.end(),
                    [](const BBoxData &b) { return !b.valid; }),
      slots.end());

  }

  void MASTER::set_thermal()
  {
    thermal_mat = original_thermal->clone();
    cv::resize(thermal_mat, thermal_mat, cv::Size(640, 480), 0, 0, cv::INTER_CUBIC);
    delete original_thermal;
    isRecv_thermal = false;
  }

  void MASTER::update()
  {
    // clone_mat = original->clone();
    // cv::resize(clone_mat, clone_mat, cv::Size(640, 480), 0, 0, cv::INTER_CUBIC);

    yolo_input = original->clone();
    cv::resize(yolo_input, yolo_input, cv::Size(640, 480));

    clone_mat = yolo_input.clone(); // 시각화용
    cv::rectangle(clone_mat, cv::Rect(0, 0, 150, 50), cv::Scalar(255, 255, 255), cv::FILLED, 8);
    cv::rectangle(clone_mat, cv::Rect(0, 51, 150, 50), cv::Scalar(255, 255, 255), cv::FILLED, 8);
    cv::rectangle(clone_mat, cv::Rect(0, 101, 150, 50), cv::Scalar(255, 255, 255), cv::FILLED, 8);
    cv::rectangle(clone_mat, cv::Rect(640 - 200, 0, 640, 50), cv::Scalar(255, 255, 255), cv::FILLED, 8);

    switch (result_maxAngle1)
    {
    case -4:
      cv::putText(clone_mat, "left", cv::Point(0, 30), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    case -3:
      cv::putText(clone_mat, "bottom_left", cv::Point(0, 30), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    case -2:
      cv::putText(clone_mat, "bottom", cv::Point(0, 30), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    case -1:
      cv::putText(clone_mat, "bottom_right", cv::Point(0, 30), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    case 0:
      cv::putText(clone_mat, "right", cv::Point(0, 30), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    case 1:
      cv::putText(clone_mat, "top_right", cv::Point(0, 30), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    case 2:
      cv::putText(clone_mat, "top", cv::Point(0, 30), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    case 3:
      cv::putText(clone_mat, "top_left", cv::Point(0, 30), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    case 4:
      cv::putText(clone_mat, "left", cv::Point(0, 30), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    default:
      break;
    }

    switch (result_maxAngle2)
    {
    case -4:
      cv::putText(clone_mat, "left", cv::Point(0, 81), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    case -3:
      cv::putText(clone_mat, "bottom_left", cv::Point(0, 81), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    case -2:
      cv::putText(clone_mat, "bottom", cv::Point(0, 81), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    case -1:
      cv::putText(clone_mat, "bottom_right", cv::Point(0, 81), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    case 0:
      cv::putText(clone_mat, "right", cv::Point(0, 81), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    case 1:
      cv::putText(clone_mat, "top_right", cv::Point(0, 81), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    case 2:
      cv::putText(clone_mat, "top", cv::Point(0, 81), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    case 3:
      cv::putText(clone_mat, "top_left", cv::Point(0, 81), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    case 4:
      cv::putText(clone_mat, "left", cv::Point(0, 81), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    default:
      break;
    }

    switch (result_maxAngle3)
    {
    case -4:
      cv::putText(clone_mat, "left", cv::Point(0, 131), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    case -3:
      cv::putText(clone_mat, "bottom_left", cv::Point(0, 131), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    case -2:
      cv::putText(clone_mat, "bottom", cv::Point(0, 131), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    case -1:
      cv::putText(clone_mat, "bottom_right", cv::Point(0, 131), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    case 0:
      cv::putText(clone_mat, "right", cv::Point(0, 131), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    case 1:
      cv::putText(clone_mat, "top_right", cv::Point(0, 131), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    case 2:
      cv::putText(clone_mat, "top", cv::Point(0, 131), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    case 3:
      cv::putText(clone_mat, "top_left", cv::Point(0, 131), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    case 4:
      cv::putText(clone_mat, "left", cv::Point(0, 131), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    default:
      break;
    }
    switch (c_rotation_state)
    {
    case 1:
      cv::putText(clone_mat, "CW", cv::Point(640 - 200, 30), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    case 2:
      cv::putText(clone_mat, "CCW", cv::Point(640 - 200, 30), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    case 0:
      cv::putText(clone_mat, "   ", cv::Point(640 - 200, 30), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    default:
      break;
    }
    switch (direction_rotation)
    {
    case 1:
      cv::putText(clone_mat, "CW", cv::Point(640 - 200, 30), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    case 0:
      cv::putText(clone_mat, "CCW", cv::Point(640 - 200, 30), 0.5, 1, cv::Scalar(0, 0, 0), 2, 8);
      break;
    default:
      break;
    }

    qr_setting();                         
    hazmat_setting(yolo_input);
    if (circle_c_.empty() || !circle_c_[0].valid) RCLCPP_WARN(this->get_logger(), "⚠️ No valid C detected — skip c_basic_setting()");
    else if (c_rotation_state != 0) RCLCPP_WARN(this->get_logger(), "⚠️ C is rotating (state=%d), skip c_basic_setting()", c_rotation_state);
    else c_basic_setting();

    detect_square_rotation();

    if (!info.empty() && qr_flag)
    {
      cv::Size text_size = cv::getTextSize(info, cv::FONT_HERSHEY_SIMPLEX, 1, 2, nullptr);
      cv::Point2i bg_top_left(points[0].x / 2, points[0].y - text_size.height - 20);
      cv::Point2i bg_bottom_right(bg_top_left.x + text_size.width, bg_top_left.y + text_size.height + 5);

      cv::Rect qr_rect(bg_top_left, bg_bottom_right);
      qr_boxes.push_back(qr_rect);

      cv::rectangle(clone_mat, bg_top_left, bg_bottom_right, cv::Scalar(0, 0, 0), -1); // -1은 사각형을 채우라는 의미
      cv::putText(clone_mat, info, cv::Point(points[0].x / 2, points[0].y - 20), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
      cv::polylines(clone_mat, points, true, cv::Scalar(0, 0, 0), 5);
    }

    // {
    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", clone_mat).toImageMsg();
    img_result->publish(*msg);
    // }

    delete original;
    isRecv = false;
  }
  // =========================================================================================
  void MASTER::hazmat_setting(const cv::Mat& input)
  {
      auto output_names = net.getUnconnectedOutLayersNames();
      std::vector<cv::Mat> detections;

      cv::dnn::blobFromImage(input, blob, 0.00392,
                              cv::Size(416, 416),
                              cv::Scalar(), true, false, CV_32F);
      net.setInput(blob);
      net.forward(detections, output_names);

      std::vector<int> indices[NUM_CLASSES];
      std::vector<cv::Rect> boxes[NUM_CLASSES];
      std::vector<float> scores[NUM_CLASSES];

      // Detection 결과 처리
      for (auto &output : detections)
      {
          const int num_boxes = output.rows;
          for (int i = 0; i < num_boxes; i++)
          {
              auto x = output.at<float>(i, 0) * input.cols;
              auto y = output.at<float>(i, 1) * input.rows;
              auto width  = output.at<float>(i, 2) * input.cols;
              auto height = output.at<float>(i, 3) * input.rows;
              cv::Rect rect(x - width/2, y - height/2, width, height);

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

      // 클래스별 NMS
      for (int c = 0; c < NUM_CLASSES; c++)
          cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);

      // 박스 그리기
      for (int c = 0; c < NUM_CLASSES; c++)
      {
          for (size_t i = 0; i < indices[c].size(); ++i)
          {
              const auto &rect = boxes[c][indices[c][i]];
              const auto color = colors[c % NUM_COLORS];
              bool isOverlapping = false;

              // --- QR 박스와 겹치는지 확인 (IoU 기반, 작든 크든 조금만 겹쳐도 제외) ---
              for (const auto &qr_rect : qr_boxes)
              {
                  float interArea = (rect & qr_rect).area();
                  float overlapRatio = interArea / std::min(rect.area(), qr_rect.area()); 
                  if (overlapRatio > 0.0f) // 조금이라도 겹치면 제외
                  {
                      isOverlapping = true;
                      break;
                  }
              }

              // circle_c와 겹치는지 확인
              if (!isOverlapping)
              {
                  for (const auto &circle_box : circle_c_)
                  {
                      if (circle_box.valid)
                      {
                          cv::Rect box_rect(circle_box.x1, circle_box.y1,
                                            circle_box.x2 - circle_box.x1,
                                            circle_box.y2 - circle_box.y1);
                          if (isRectOverlapping(rect, box_rect))
                          {
                              isOverlapping = true;
                              break;
                          }
                      }
                  }
              }

              // circle_square와 겹치는지 확인
              if (!isOverlapping)
              {
                  for (const auto &sq_box : circle_square_)
                  {
                      if (sq_box.valid)
                      {
                          cv::Rect box_rect(sq_box.x1, sq_box.y1,
                                            sq_box.x2 - sq_box.x1,
                                            sq_box.y2 - sq_box.y1);
                          if (isRectOverlapping(rect, box_rect))
                          {
                              isOverlapping = true;
                              break;
                          }
                      }
                  }
              }

              // 겹치지 않은 박스만 그림
              if (!isOverlapping)
              {
                  cv::rectangle(clone_mat, rect, color, 3);

                  std::ostringstream label_ss;
                  label_ss << class_names[c] << ": " << std::fixed << std::setprecision(2) << scores[c][indices[c][i]];
                  auto label = label_ss.str();

                  int baseline;
                  auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);

                  int label_x = rect.x;
                  if (label_x + label_bg_sz.width > clone_mat.cols) label_x = clone_mat.cols - label_bg_sz.width - 10;
                  if (label_x < 0) label_x = 10;

                  if ((rect.y - label_bg_sz.height - baseline - 10) >= 0)
                  {
                      cv::rectangle(clone_mat, cv::Point(label_x, rect.y - label_bg_sz.height - baseline - 10),
                                    cv::Point(label_x + label_bg_sz.width, rect.y), cv::Scalar(255, 255, 255), cv::FILLED);
                      cv::putText(clone_mat, label.c_str(), cv::Point(label_x, rect.y - baseline - 5),
                                  cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
                  }
                  else
                  {
                      cv::rectangle(clone_mat, cv::Point(label_x, rect.y + rect.height),
                                    cv::Point(label_x + label_bg_sz.width, rect.y + rect.height + label_bg_sz.height + baseline + 10),
                                    cv::Scalar(255, 255, 255), cv::FILLED);
                      cv::putText(clone_mat, label.c_str(), cv::Point(label_x, rect.y + rect.height + baseline + 5),
                                  cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
                  }
              }
          }
      }
  }


  bool MASTER::isRectOverlapping(const cv::Rect &rect1, const cv::Rect &rect2)
  {
      return (rect1 & rect2).area() > 0; // 교집합이 1픽셀이라도 있으면 true
  }


  void MASTER::qr_setting()
  {
    output_qr = clone_mat.clone();
    cv::cvtColor(clone_mat, gray_clone, COLOR_BGR2GRAY);

    if (detector.detect(gray_clone, points))
    {
      info = detector.decode(gray_clone, points);
      qr_flag = true;

      if (!info.empty())
      {
        cv::Size text_size = cv::getTextSize(info, cv::FONT_HERSHEY_SIMPLEX, 1, 2, nullptr);
        cv::Point2i bg_top_left(points[0].x / 2, points[0].y - text_size.height - 20);
        cv::Point2i bg_bottom_right(bg_top_left.x + text_size.width, bg_top_left.y + text_size.height + 5);

        cv::Rect qr_rect(bg_top_left, bg_bottom_right);
        qr_boxes.push_back(qr_rect);

        cv::rectangle(output_qr, bg_top_left, bg_bottom_right, cv::Scalar(0, 0, 0), -1); // -1은 사각형을 채우라는 의미
        cv::putText(output_qr, info, cv::Point(points[0].x / 2, points[0].y - 20), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        cv::polylines(output_qr, points, true, cv::Scalar(0, 0, 0), 5);
      }
      // if (qr_flag == true)
      // {
      //   auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", output_qr).toImageMsg();
      //   img_result->publish(*msg);
      // }
    }
    else
    {
      qr_flag = false;
    }
  }

  void MASTER::c_basic_setting()
  {
    // 출력용 복제 이미지
    output_c = yolo_input.clone();

    // ROI별 중심 좌표 저장용 (최대 2개 가정)
    static std::vector<cv::Point> prev_centers(2, cv::Point(-1, -1));

    // circle_c_ : YOLO가 검출한 원형 영역들
    for (size_t i = 0; i < circle_c_.size(); ++i)
    {
      const auto &bbox = circle_c_[i];
      if (!bbox.valid) continue;

      // ROI 좌표 계산
      int x = std::max(0, (int)bbox.x1);
      int y = std::max(0, (int)bbox.y1);
      int w = std::min((int)(bbox.x2 - bbox.x1), clone_mat.cols - x);
      int h = std::min((int)(bbox.y2 - bbox.y1), clone_mat.rows - y);
      
      if (x < 0 || y < 0) continue;
      if (x + w > clone_mat.cols) w = clone_mat.cols - x;
      if (y + h > clone_mat.rows) h = clone_mat.rows - y;
      if (w <= 0 || h <= 0) continue;


      // ROI 추출
      C_roi = clone_mat(cv::Rect(x, y, w, h)).clone();

      // ROI를 정사각형 비율로 정규화 (로컬 내에서만)
      if (w < h)
      {
        cv::resize(C_roi, C_roi, cv::Size(h, h));
        w = h;
      }
      else if (h < w)
      {
        cv::resize(C_roi, C_roi, cv::Size(w, w));
        h = w;
      }

      // 전처리
      cv::cvtColor(C_roi, gray_clone, cv::COLOR_BGR2GRAY);
      cv::GaussianBlur(gray_clone, gray_clone, cv::Size(5, 5), 0);
      cv::threshold(gray_clone, gray_clone, 100, 255, cv::THRESH_BINARY_INV);

      // 컨투어 검출
      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(gray_clone, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
      if (contours.empty()) return;

      // 면적 기준 정렬 (큰 컨투어 → 작은 컨투어)
      std::sort(contours.begin(), contours.end(),
                [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                  return cv::contourArea(a) > cv::contourArea(b);
                });

      // 상위 3개만 사용
      int numContours = std::min(3, (int)contours.size());

      // 색상 (빨, 초, 파)
      std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 0, 255),
        cv::Scalar(0, 255, 0),
        cv::Scalar(255, 0, 0)
      };

      // ROI 중심 (ROI 내부 기준)
      cv::Point center(w / 2, h / 2);

      // 시각화
      cv::Mat debug_img = C_roi.clone();

      std::vector<float> radius_candidates;
      radius_candidates.reserve(numContours);

      for (int i = 0; i < numContours; i++)
      {
        // 컨투어 내부 색칠
        cv::drawContours(debug_img, contours, i, colors[i], -1);

        // 중심 기준 가장 먼 점 → 반지름 계산
        std::vector<double> dists;
        dists.reserve(contours[i].size());
        for (const auto &p : contours[i])
        {
          dists.push_back(cv::norm(center - p));
        }

        std::sort(dists.begin(), dists.end());

        double maxDist = dists[dists.size() / 2];
        maxDist = std::min(maxDist,
                          (double)std::min(C_roi.cols, C_roi.rows) / 2);

        radius_candidates.push_back((float)maxDist);

        // 시각화는 유지
        cv::circle(debug_img, center, (int)maxDist, colors[i], 2);
      }

      if (!radius_candidates.empty())
      {
        std::sort(radius_candidates.begin(), radius_candidates.end());

        float median_r = radius_candidates[radius_candidates.size() / 2];

        // ✅ circles에는 단 하나만!
        circles.clear();
        circles.push_back(cv::Vec3f(center.x, center.y, median_r));

        RCLCPP_INFO(this->get_logger(),
          "Selected median C radius: %.2f", median_r);
      }

      // 로그 표시
      // cv::imshow("Top 3 Filled Contours", debug_img);
      // cv::waitKey(1);

      cv::Point &prev_center = prev_centers[i < prev_centers.size() ? i : 0];

      // 프레임 기준 후보
      bool frame_found = false;
      cv::Point frame_center(-1, -1);
      int frame_radius = 0;

      int roi_min_r = std::min(w, h) * 0.25;
      int roi_max_r = std::min(w, h) * 0.6;

      for (size_t j = 0; j < circles.size(); j++)
      {
        // 로컬 고정
        const cv::Vec3f local_c = circles[j];

        // RCLCPP_INFO(this->get_logger(),
        //   "Circle data: x=%.2f, y=%.2f, r=%.2f",
        //   local_c[0], local_c[1], local_c[2]);

        cv::Point center(cvRound(local_c[0]), cvRound(local_c[1]));
        int two_radius = cvRound(local_c[2]);

        // 흔들림 제거
        if (prev_center.x != -1)
        {
          float dist = cv::norm(center - prev_center);
          if (dist > 10.0)
          {
            prev_center = center;
            continue;
          }
        }
        prev_center = center;

        // ROI 기준 완전 포함 검사 (clone_mat ❌)
        if (!(center.x - two_radius >= 0 &&
              center.y - two_radius >= 0 &&
              center.x + two_radius < C_roi.cols &&
              center.y + two_radius < C_roi.rows))
        {
          continue;
        }

        // circle_shape용 전역 c는 여기서만 세팅
        c = local_c;
        circle_shape(two_radius);

        RCLCPP_INFO(this->get_logger(),
          "Two shape check result: %s",
          find_two ? "Found" : "Not Found");

        if (find_two)
        {
          // 프레임 후보 선택 (중간 사이즈 우선)
          if (!frame_found)
          {
            frame_center = center;
            frame_radius = two_radius;
            frame_found = true;
          }
          else
          {
            // 더 적절한 반지름이면 교체 (ex: 중간값 기준)
            if (two_radius > roi_min_r && two_radius < roi_max_r)
            {
              frame_center = center;
              frame_radius = two_radius;
              frame_found = true;
            }
          }

          // 시각화는 그대로 OK
          cv::Point global_center(center.x + x, center.y + y);
          cv::circle(output_c, global_center,
                    two_radius, cv::Scalar(0, 255, 0), 2);
        }
      }

      if (frame_found)
      {
        bool same_target = false;

        if (best_center.x != -1)
        {
          float dist = cv::norm(frame_center - best_center);
          float dr   = std::abs(frame_radius - best_radius);

          if (dist < 15.0 && dr < best_radius * 0.2)
            same_target = true;
        }

        if (same_target)
        {
          best_stable_count++;
        }
        else
        {
          best_center = frame_center;
          best_radius = frame_radius;
          best_stable_count = 1;
        }

        if (c_locked) {
          find_c_123(best_radius);
          break;
        }

        RCLCPP_INFO(this->get_logger(),
          "Best C stable (frame): %d / %d",
          best_stable_count, REQUIRED_STABLE);

        if (best_stable_count >= REQUIRED_STABLE && !c_locked)
        {
          c_locked = true;

          c = cv::Vec3f(best_center.x, best_center.y, best_radius);

          RCLCPP_INFO(this->get_logger(),
            "C STABLE LOCKED at (%d, %d), r=%d",
            best_center.x, best_center.y, best_radius);

          find_c_123(best_radius);
        }

      }
      else
      {
        if (!c_locked)
        {
          best_stable_count = 0;
          best_center = cv::Point(-1, -1);
          best_radius = 0;

          c_rotation_started = false;
          c_stable_count = 0;
          first_c_latched = false;

          RCLCPP_INFO(this->get_logger(),
            "C lost, reset stability");
        }
      }


      circles.clear();
    }
  }

  // C 모양
  void MASTER::circle_shape(int two_radius)
  {
      int cx = (int)c[0];
      int cy = (int)c[1];

      int y_start = std::max(0, cy - two_radius);
      int y_end   = std::min(C_roi.rows, cy + two_radius);
      int x_start = std::max(0, cx - two_radius);
      int x_end   = std::min(C_roi.cols, cx + two_radius);

      // ✅ OpenCV assert 방지 (핵심)
      if (x_end <= x_start || y_end <= y_start)
      {
        find_two = false;
        return;
      }

      two_mat = C_roi(cv::Rect(
        x_start, y_start,
        x_end - x_start,
        y_end - y_start)).clone();
      
      cv::resize(two_mat, two_mat, cv::Size(300, 300), 0, 0, cv::INTER_CUBIC);
      cv::cvtColor(two_mat, two_gray, cv::COLOR_BGR2GRAY);
      cv::threshold(two_gray, two_binary, 80, 255, cv::THRESH_BINARY);
      two_binary = ~two_binary;
      find_two = check_black(two_binary);
      // cv::imshow("Two Binary", two_mat);
      // cv::waitKey(1);
  }

  // 첫번째 C 부터 순서대로 찾게 하는 코드
  void MASTER::find_c_123(int two_radius)
  {
    if (C_roi.empty())
    {
      RCLCPP_WARN(this->get_logger(), "C_roi is empty.");
      return;
    }

    int cx = static_cast<int>(c[0]);
    int cy = static_cast<int>(c[1]);

    cv::Rect imgRect(0, 0, C_roi.cols, C_roi.rows);

    /* ===================== 1번 C ===================== */
    int one_radius = static_cast<int>(2.3 * two_radius);

    cv::Rect roi1(
        cx - one_radius,
        cy - one_radius,
        one_radius * 2,
        one_radius * 2
    );

    roi1 = roi1 & imgRect;

    if (roi1.width <= 0 || roi1.height <= 0)
    {
      first_c_latched = false;
      c_stable_count = 0;
      return;
    }

    one_mat = C_roi(roi1).clone();

    cv::resize(one_mat, one_mat, cv::Size(300, 300));
    cv::cvtColor(one_mat, one_gray, cv::COLOR_BGR2GRAY);
    cv::threshold(one_gray, one_binary, 80, 255, cv::THRESH_BINARY);
    one_binary = ~one_binary;

    bool one_found = check_black(one_binary);

    /* ===================== 3번 C ===================== */
    int three_radius = static_cast<int>(0.4167 * two_radius);

    cv::Rect roi3(
        cx - three_radius,
        cy - three_radius,
        three_radius * 2,
        three_radius * 2
    );

    roi3 = roi3 & imgRect;

    if (roi3.width <= 0 || roi3.height <= 0)
      return;

    three_mat = C_roi(roi3).clone();

    cv::resize(three_mat, three_mat, cv::Size(300, 300));
    cv::cvtColor(three_mat, three_gray, cv::COLOR_BGR2GRAY);
    cv::threshold(three_gray, three_binary, 80, 255, cv::THRESH_BINARY);
    three_binary = ~three_binary;

    /* ===================== 디버깅용 ROI 시각화 ===================== */

    // cv::Mat debug_img = C_roi.clone();

    // // 1번 ROI (빨강)
    // if (roi1.width > 0 && roi1.height > 0)
    // {
    //     cv::rectangle(debug_img, roi1, cv::Scalar(0, 0, 255), 2);
    // }

    // // 2번 ROI (two_radius 기준 원 영역, 초록)
    // cv::Rect roi2(
    //     cx - two_radius,
    //     cy - two_radius,
    //     two_radius * 2,
    //     two_radius * 2
    // );
    // roi2 = roi2 & imgRect;

    // if (roi2.width > 0 && roi2.height > 0)
    // {
    //     cv::rectangle(debug_img, roi2, cv::Scalar(0, 255, 0), 2);
    // }

    // // 3번 ROI (파랑)
    // if (roi3.width > 0 && roi3.height > 0)
    // {
    //     cv::rectangle(debug_img, roi3, cv::Scalar(255, 0, 0), 2);
    // }

    // // 중심점 표시 (노랑)
    // cv::circle(debug_img, cv::Point(cx, cy), 3, cv::Scalar(0,255,255), -1);

    // // 디버그 창
    // cv::imshow("C ROI Debug", debug_img);
    // cv::waitKey(1);

    /* ===================== 안정화 로직 ===================== */
    if (one_found)
    {
      if (!first_c_latched)
        RCLCPP_INFO(this->get_logger(), "First C detected.");

      first_c_latched = true;
      c_stable_count++;
    }
    else
    {
      first_c_latched = false;
      c_stable_count = 0;
    }

    if (first_c_latched && c_stable_count >= C_STABLE_TH)
    {
      if (!c_rotation_started)
      {
        RCLCPP_INFO(this->get_logger(),
          "C stable. Starting circle rotation detection.");

        angleFrequency1.clear();
        angleFrequency2.clear();
        angleFrequency3.clear();
        movement_count = 0;

        c_rotation_started = true;
      }

      detect_circle_rotation();
    }
  }

  void MASTER::detect_circle_rotation()
  {
    // first_circle
    double sumAngles1 = 0.0;
    int count1 = 0;
    int radius1 = 125; // 원의 반지름
    double temp_radian1 = 0;
    double angleRadians1 = 0;

    for (int y = 0; y < one_binary.rows; y++)
    {
      for (int x = 0; x < one_binary.cols; x++)
      {
        if (one_binary.at<uchar>(y, x) == 0)
        {
          double distance1 = std::sqrt(std::pow(x - 150, 2) + std::pow(y - 150, 2));
          if (std::abs(distance1 - radius1) < 1.0)
          {
            angleRadians1 = std::atan2(y - 150, x - 150) * 180.0 / CV_PI;
            if (count1 > 0 && abs(sumAngles1 / count1 - angleRadians1) > 180)
            {
              angleRadians1 -= 360;
            }
            sumAngles1 += angleRadians1;
            count1++;
          }
        }
      }
      if (abs(temp_radian1 - angleRadians1) > 180)
        break;
    }

    double averageAngle1 = 0;
    if (count1 > 0)
    {
      averageAngle1 = -(sumAngles1 / count1);
      if (averageAngle1 > 180)
        averageAngle1 -= 360;
      else if (averageAngle1 < -180)
        averageAngle1 += 360;
    }

    // ROI 체크를 위한 코드는 유지
    cv::Point cen(150 + radius1 * cos(averageAngle1 / 180.0 * PI), 150 + radius1 * sin(-averageAngle1 / 180.0 * PI));
    int radius_roi = 20;
    cv::Rect r(cen.x - radius_roi, cen.y - radius_roi,
           radius_roi * 2, radius_roi * 2);

    if (!safe_rect(one_binary, r))
      return;

    cv::Mat roi = one_binary(r);

    cv::Mat mask(roi.size(), roi.type(), Scalar::all(0));
    cv::circle(mask, Point(radius_roi, radius_roi), radius_roi, Scalar::all(255), -1);
    Mat eye_cropped = roi & mask;
    Scalar aver = mean(eye_cropped);
    if (aver[0] < 5 || 200 < aver[0])
      return;

    // second_circle
    double sumAngles2 = 0.0;
    int count2 = 0;
    int radius2 = 125; // 원의 반지름
    double temp_radian2 = 0;
    double angleRadians2 = 0;

    for (int y = 0; y < two_binary.rows; y++)
    {
      for (int x = 0; x < two_binary.cols; x++)
      {
        if (two_binary.at<uchar>(y, x) == 0)
        {
          // 중심 좌표로부터의 거리 계산
          double distance2 = std::sqrt(std::pow(x - 150, 2) + std::pow(y - 150, 2));

          if (std::abs(distance2 - radius2) < 1.0) // 거리가 125인 지점 판단
          {
            angleRadians2 = std::atan2(y - 150, x - 150) * 180.0 / CV_PI;
            if (count2 > 0 && abs(sumAngles2 / count2 - angleRadians2) > 180)
            {
              angleRadians2 -= 360;
            }
            sumAngles2 += angleRadians2;
            count2++;
          }
        }
      }
      if (abs(temp_radian2 - angleRadians2) > 180)
        break;
    }

    double averageAngle2 = 0;
    if (count2 > 0)
    {
      averageAngle2 = -(sumAngles2 / count2);
      if (averageAngle2 > 180)
      {
        averageAngle2 -= 360;
      }
      else if (averageAngle2 < -180)
      {
        averageAngle2 += 360;
      }
    }

    cv::Point cen2(150 + radius2 * cos(averageAngle2 / 180.0 * PI), 150 + radius2 * sin(-averageAngle2 / 180.0 * PI));
    int radius_roi2 = 10;
    cv::Rect r2(cen2.x - radius_roi2, cen2.y - radius_roi2,
            radius_roi2 * 2, radius_roi2 * 2);

    if (!safe_rect(two_binary, r2))
      return;

    cv::Mat roi2 = two_binary(r2);


    cv::Mat mask2(roi2.size(), roi2.type(), Scalar::all(0));

    cv::circle(mask2, Point(radius_roi2, radius_roi2), radius_roi2, Scalar::all(255), -1);

    Mat eye_cropped2 = roi2 & mask2;

    // third_circle
    double sumAngles3 = 0.0;
    int count3 = 0;
    int radius3 = 125; // 원의 반지름
    double temp_radian3 = 0;
    double angleRadians3 = 0;

    for (int y = 0; y < three_binary.rows; y++)
    {
      for (int x = 0; x < three_binary.cols; x++)
      {
        if (three_binary.at<uchar>(y, x) == 0)
        {
          // 중심 좌표로부터의 거리 계산
          double distance3 = std::sqrt(std::pow(x - 150, 2) + std::pow(y - 150, 2));
          if (std::abs(distance3 - radius3) < 1.0) // 거리가 125인 지점 판단
          {
            angleRadians3 = std::atan2(y - 150, x - 150) * 180.0 / CV_PI;
            if (count3 > 0 && abs(sumAngles3 / count3 - angleRadians3) > 180)
            {
              angleRadians3 -= 360;
            }
            sumAngles3 += angleRadians3;
            count3++;
          }
        }
      }
      if (abs(temp_radian3 - angleRadians3) > 180)
        break;
    }

    double averageAngle3 = 0;
    if (count3 > 0)
    {
      averageAngle3 = -(sumAngles3 / count3);
      if (averageAngle3 > 180)
      {
        averageAngle3 -= 360;
      }
      else if (averageAngle3 < -180)
      {
        averageAngle3 += 360;
      }
    }
    int averageAngle1_i = averageAngle1;
    int averageAngle2_i = averageAngle2;
    int averageAngle3_i = averageAngle3;

    averageAngle_1 = (averageAngle1_i + 22.5 * ((averageAngle1_i > 0) ? 1 : -1)) / 45.0;
    averageAngle_2 = (averageAngle2_i + 22.5 * ((averageAngle2_i > 0) ? 1 : -1)) / 45.0;
    averageAngle_3 = (averageAngle3_i + 22.5 * ((averageAngle3_i > 0) ? 1 : -1)) / 45.0;

    if (averageAngle_1 == -4)
      averageAngle_1 = 4;
    if (averageAngle_2 == -4)
      averageAngle_2 = 4;
    if (averageAngle_3 == -4)
      averageAngle_3 = 4;

    if (movement_count <= 21)
      movement_count += 1;

    if (movement_count < 20)
    {
      angleFrequency1[averageAngle_1]++;
      if (angleFrequency1[averageAngle_1] > maxFrequency1)
      {
        maxFrequency1 = angleFrequency1[averageAngle_1];
        maxFrequencyAngle1 = averageAngle_1;
      }
      angleFrequency2[averageAngle_2]++;
      if (angleFrequency2[averageAngle_2] > maxFrequency2)
      {
        maxFrequency2 = angleFrequency2[averageAngle_2];
        maxFrequencyAngle2 = averageAngle_2;
      }
      angleFrequency3[averageAngle_3]++;
      if (angleFrequency3[averageAngle_3] > maxFrequency3)
      {
        maxFrequency3 = angleFrequency3[averageAngle_3];
        maxFrequencyAngle3 = averageAngle_3;
      }
    }

    // if 문 시작 조건 로그
    RCLCPP_INFO(this->get_logger(),
      "movement_count: %d, c_dir_flag: %s, rotation_enabled: %s",
      movement_count,
      c_dir_flag ? "true" : "false",
      rotation_enabled ? "true" : "false"); 

    // 방향 검출 로직
    if (movement_count >= 20 && !c_dir_flag)
    {
      // 안정화된 3개 원 기준 변화량
      int diff1 = averageAngle_1 - result_maxAngle1;
      int diff2 = averageAngle_2 - result_maxAngle2;
      int diff3 = averageAngle_3 - result_maxAngle3;
      
      RCLCPP_INFO(this->get_logger(),
        "Angle Diffs: d1=%d, d2=%d, d3=%d",
        diff1, diff2, diff3);

      if (!result_maxAngle1 && !result_maxAngle2 && !result_maxAngle3)
      {
        result_maxAngle1 = maxFrequencyAngle1;
        result_maxAngle2 = maxFrequencyAngle2;
        result_maxAngle3 = maxFrequencyAngle3;

        RCLCPP_INFO(this->get_logger(),
          "Initial max angles set: %d, %d, %d",
          result_maxAngle1,
          result_maxAngle2,
          result_maxAngle3);

        return;
      }
      int sumDiff = diff1 + diff2 + diff3;

      // wrap 보정
      if (sumDiff > 4)  sumDiff -= 8;
      if (sumDiff < -4) sumDiff += 8;

      // if (abs(sumDiff) >= 2 && square_color == 0)   // threshold
      // {
      //   direction_rotation = (sumDiff > 0) ? 1 : 0; // 1=CW, 0=CCW
      //   c_dir_flag = true;
      //   c_direction_found = true;

      //   RCLCPP_INFO(this->get_logger(),
      //     "Rotation detected (%s) | diff=%d",
      //     direction_rotation ? "CW" : "CCW",
      //     sumDiff);
      // }
    } else 
    RCLCPP_INFO(this->get_logger(),
      "Collecting data for direction... %d/20",
      movement_count);

  }

  void MASTER::detect_square_rotation()
  {
    // YOLO로 박스 감지 안 됨 → 안정화 리셋
    if (square_boxes.empty()) {
      stabilize_count = 0;
      rotation_enabled = false;
      if (log_once)
        RCLCPP_INFO(this->get_logger(), "No YOLO bounding boxes for square detected");
      log_once = false;
      return;
    }
    log_once = true;

    // 안정화 카운트 증가
    if (!rotation_enabled) {
      stabilize_count++;
      RCLCPP_INFO(this->get_logger(), "Stabilizing... %d/%d", stabilize_count, STABILIZE_LIMIT);

      if (stabilize_count >= STABILIZE_LIMIT) {
        rotation_enabled = true;
        RCLCPP_INFO(this->get_logger(), "Ready to detect rotation!");
      }
      return; // 안정화 중이면 리턴
    }

    // clone_mat
    cv::Mat gray;
    cv::cvtColor(yolo_input, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

    // YOLO 감지된 square 박스 2개 중 유효한 것만 처리
    for (const auto &bbox : square_boxes)
    {
      if (!bbox.valid) continue;

      // YOLO에서 받은 박스 기준 ROI 설정
      int x = std::max(0, (int)bbox.x1);
      int y = std::max(0, (int)bbox.y1);
      int w = std::min((int)(bbox.x2 - bbox.x1), gray.cols - x);
      int h = std::min((int)(bbox.y2 - bbox.y1), gray.rows - y);

      if (w <= 0 || h <= 0)
          continue;

      cv::Rect roi_rect(x, y, w, h);
      cv::Mat roi = gray(roi_rect);
      cv::Mat binary;
      cv::Mat binary_not_adapt;

      double thresh_value = 100;  // 임계값 (0~255)
      if (square_color == 1) { // 흰색 바탕
        cv::threshold(roi, binary, thresh_value, 255, cv::THRESH_BINARY_INV);
      }
      else { // 검은색 바탕
        cv::threshold(roi, binary, thresh_value, 255, cv::THRESH_BINARY);
      }

      // cv::adaptiveThreshold(roi, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
      //                       cv::THRESH_BINARY_INV, 11, 2);

      // cv::imshow("Contours Debug", roi);
      // cv::imshow("bbox Debug", binary);
      // cv::waitKey(1);
      

      cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
      cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
      cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);

      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(binary, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

      // // === 디버그용 GUI 창 ===
      // cv::Mat display;
      // cv::cvtColor(binary, display, cv::COLOR_GRAY2BGR);
      // for (const auto &contour : contours)
      // {
      //     double area = cv::contourArea(contour);
      //     if (area < (w*h)*0.005 || area > (w*h)*0.5 || area < 800.0)
      //         continue;

      //     std::vector<cv::Point> approx;
      //     cv::approxPolyDP(contour, approx, cv::arcLength(contour, true)*0.05, true);

      //     if (approx.size() >= 4 && approx.size() <= 8)
      //     {
      //         cv::Moments m = cv::moments(contour);
      //         if (m.m00 == 0) continue;
      //         cv::Point2f center(m.m10/m.m00, m.m01/m.m00);

      //         cv::drawContours(display, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(0,255,0), cv::FILLED);
      //         cv::circle(display, center, 3, cv::Scalar(0,0,255), -1);
      //     }
      // }

      double min_area = (w * h) * 0.005;
      double max_area = (w * h) * 0.5;

      // 1. 유효한 컨투어만 모으기
      std::vector<std::pair<std::vector<cv::Point>, double>> validContours;

      for (const auto &contour : contours)
      {
        double area = cv::contourArea(contour);
        if (area < min_area || area > max_area)
          continue;

        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, cv::arcLength(contour, true) * 0.05, true);

        if (approx.size() >= 4 && approx.size() <= 8)
        {
          validContours.emplace_back(contour, area);
        }
      }

      // 2. 원 중심과 반지름 정의
      cv::Point2f square_center(roi.cols / 2.0f, roi.rows / 2.0f);
      float radius = std::min(roi.cols, roi.rows) / 2.0f * 0.9f;

      // 3. 원 안 최대 컨투어 찾기
      double maxArea = 0.0;
      std::vector<cv::Point> maxContour;

      for (auto &vc : validContours)
      {
        std::vector<cv::Point> &contour = vc.first;
        double area = vc.second;

        cv::Moments m = cv::moments(contour);
        if (m.m00 == 0)
          continue;

        cv::Point2f c(m.m10 / m.m00, m.m01 / m.m00);
        float dist = cv::norm(c - square_center);

        if (dist <= radius)
        {
          if (area > maxArea)
          {
            maxArea = area;
            maxContour = contour;
          }
        }
      }

      // 4. 최대 컨투어 처리
      if (!maxContour.empty())
      {
        cv::Moments m = cv::moments(maxContour);
        cv::Point2f contour_center(m.m10 / m.m00, m.m01 / m.m00);

        double angle = atan2(contour_center.y - roi.rows / 2,
                  contour_center.x - roi.cols / 2) * 180.0 / CV_PI;

        int angle_i = static_cast<int>(angle);
        int quantized_angle = (angle_i + 22.5 * ((angle_i > 0) ? 1 : -1)) / 45.0;
        if (quantized_angle == -4) quantized_angle = 4;

        if (movement_count_square < 20)
        {
          angleFrequency[quantized_angle + 4]++;
          if (angleFrequency[quantized_angle + 4] > maxFrequency)
          {
            maxFrequency = angleFrequency[quantized_angle + 4];
            maxFrequencyAngle = quantized_angle;
          }
          movement_count_square++;
        }
        else if (movement_count_square == 20)
        {
          result_maxAngle = maxFrequencyAngle;
          first_square_angle = quantized_angle;

          if (first_square_angle == 4 || first_square_angle == -3 ||
            first_square_angle == maxFrequencyAngle)
          {
            movement_count_square = 15;
            RCLCPP_INFO(this->get_logger(), "Reset for stable detection");
          }
          else
          {
            square_direction_found = true;
            RCLCPP_INFO(this->get_logger(), "Square direction found!");
            movement_count_square++;
          }
        }
        else if (square_direction_found)
        {
          square_direction_found = true;
          last_square_angle = quantized_angle;
          int direction = first_square_angle - last_square_angle;

          if (last_square_angle != maxFrequencyAngle && abs(direction) > 0)
          {
            movement_count_square = false; // 한번 감지되면 멈추기
            if (direction < 0)
            {
              direction_rotation = 1;
              c_dir_flag = true;
              RCLCPP_INFO(this->get_logger(), "Detected CW rotation!");
            }
            else
            {
              direction_rotation = 0;
              c_dir_flag = true;
              RCLCPP_INFO(this->get_logger(), "Detected CCW rotation!");
            }
          }
          else {
            RCLCPP_INFO(this->get_logger(), "No change in square angle. %d to %d",
              first_square_angle, last_square_angle);
          }
        }

        draw_and_track_square(maxContour, roi_rect);
        cv::rectangle(clone_mat, roi_rect, cv::Scalar(255, 0, 0), 2);
      }
    }
  }
  
  void MASTER::draw_and_track_square(
      const std::vector<cv::Point>& contour,
      const cv::Rect& roi_rect
  )
  {
    //RCLCPP_INFO(this->get_logger(), "draw_and_track_square CALLED");

    // 회전 바운딩 박스
    cv::RotatedRect rbox = cv::minAreaRect(contour);

    // 중심점
    cv::Point2f current_center = rbox.center;

    // 추적 초기화 or 동일 객체 판단
    if (!square_tracked)
    {
      square_tracked = true;
      prev_square_center = current_center;
      last_draw_center = current_center;
    }
    else
    {
      float dist = cv::norm(current_center - prev_square_center);
      if (dist > 30.0)  // 너무 멀면 다른 객체
        //return;

      prev_square_center = current_center;
    }

    // ===== 회전 바운딩 박스 그리기 =====
    cv::Point2f vertices[4];
    rbox.points(vertices);

    for (int i = 0; i < 4; i++)
    {
      vertices[i].x += roi_rect.x;
      vertices[i].y += roi_rect.y;
    }

    for (int i = 0; i < 4; i++)
    {
      cv::line(
        clone_mat,
        vertices[i],
        vertices[(i + 1) % 4],
        cv::Scalar(0, 0, 255), 2
      );
    }

    // ===== 중심점 표시 =====
    cv::Point draw_center(
      current_center.x + roi_rect.x,
      current_center.y + roi_rect.y
    );

    cv::circle(
      clone_mat,
      draw_center,
      4,
      cv::Scalar(255, 0, 255),
      -1
    );

    // ===== 이동 궤적 표시 =====
    cv::line(
      clone_mat,
      last_draw_center + cv::Point2f(roi_rect.x, roi_rect.y),
      draw_center,
      cv::Scalar(255, 0, 255),
      2
    );

    last_draw_center = current_center;
  }


  bool MASTER::check_black(const Mat &binary_mat)
  {
    int cnt = 0;

    bool up = binary_mat.at<uchar>(20, 150);
    bool left = binary_mat.at<uchar>(150, 20);
    bool down = binary_mat.at<uchar>(280, 150);
    bool right = binary_mat.at<uchar>(150, 280);

    if (up == 1)
      cnt++;
    if (left == 1)
      cnt++;
    if (right == 1)
      cnt++;
    if (down == 1)
      cnt++;
    if (cnt >= 3)
      return true;
    else
      return false;
  }

  void updateBoxes(std::vector<BBoxData> &slots, const std::vector<BBoxData> &new_boxes)
  {
    const float DIST_THRESHOLD = 50.0f;  // 가까운 기준 거리 (픽셀 단위)

    // 일단 모두 invalid로 초기화
    for (auto &s : slots) s.valid = false;

    for (const auto &nb : new_boxes)
    {
      // 기존 슬롯 중 가장 가까운 것 찾기
      float min_dist = 1e9;
      int best_idx = -1;
      for (int i = 0; i < (int)slots.size(); ++i)
      {
        if (!slots[i].valid) continue;  // 이전 프레임에서 유효했던 것만 비교
        float d = distance(slots[i].cx, slots[i].cy, nb.cx, nb.cy);
        if (d < min_dist) {
          min_dist = d;
          best_idx = i;
        }
      }

      if (best_idx != -1 && min_dist < DIST_THRESHOLD)
      {
        // 이전과 비슷한 위치 → 갱신
        slots[best_idx] = nb;
        slots[best_idx].valid = true;
      }
      else
      {
        // 새로운 객체 → 비어있는 슬롯에 넣기
        for (auto &slot : slots)
        {
          if (!slot.valid) {
            slot = nb;
            slot.valid = true;
            break;
          }
        }
      }
    }
  }

} // namespace vision_rescue_26