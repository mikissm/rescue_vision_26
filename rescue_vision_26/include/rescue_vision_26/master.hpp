#include "rclcpp/rclcpp.hpp"
#include <string>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>

#include <iostream>
#include <vector>
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/int32.hpp"
#include <opencv2/opencv.hpp>

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <map>

#include "custom_msgs/msg/bounding_boxes.hpp"
#include "custom_msgs/msg/bounding_box.hpp"

using namespace cv;
using namespace std;

struct BBoxData {
    std::string class_name;
    float confidence;
    float x1, y1, x2, y2;
    float cx, cy;  // 중심 좌표
    bool valid = false;
};


namespace vision_rescue_26
{
  class MASTER : public rclcpp::Node
  {
  public:
    MASTER(); // ROS2에서는 argc, argv를 받지 않습니다
    ~MASTER();

    bool init();
    void run();

    // 이미지 정보
    Mat *original;
    Mat *original_thermal;
    Mat clone_mat;
    Mat thermal_mat;
    Mat gray_clone;
    bool isRecv;
    bool isRecv_thermal;
    void update();
    void set_thermal();

    // hazmat 정보
    Mat frame;
    Mat blob;
    cv::dnn::Net net;
    std::vector<std::string> class_names;
    bool isOverlapping;
    bool isRectOverlapping(const cv::Rect &rect1, const cv::Rect &rect2);
    void hazmat_setting(const cv::Mat& input);

    // qr 정보
    Mat output_qr;
    vector<Point> points;
    vector<cv::Rect> qr_boxes;
    String info;
    QRCodeDetector detector;
    void qr_setting();

    // c 정보
    bool exist_moving = false;
    Mat output_c;
    Vec3i c;
    vector<Vec3f> circles;
    int two_radius;

    Mat two_mat;
    Mat two_gray;
    Mat two_binary;
    bool find_two = false;

    Mat one_mat;
    Mat one_gray;
    Mat one_binary;
    bool find_one = false;

    Mat three_mat;
    Mat three_gray;
    Mat three_binary;

    int averageAngle_i;
    int movement_count = 0;
    int movement_count_square = 0;
    int previous_value = 0;
    int count_same_value = 0;
    int first_i = 0;
    int last_i = 0;
    int direction = 0;
    int direction_rotation = 2;
    std::map<int, int> angleFrequency1;
    std::map<int, int> angleFrequency2;
    std::map<int, int> angleFrequency3;
    int maxFrequencyAngle1 = 0;
    int maxFrequencyAngle2 = 0;
    int maxFrequencyAngle3 = 0;
    int maxFrequency1 = 0;
    int maxFrequency2 = 0;
    int maxFrequency3 = 0;
    int result_maxAngle1;
    int result_maxAngle2;
    int result_maxAngle3;

    void circle_shape(int radius);
    void find_c_123(int two_radius);
    void detect_circle_rotation();
    bool check_black(const Mat &binary_mat);
    void c_basic_setting();
    void detect_square_rotation();
    double prev_rotation_angle;
    bool first_rotation_detect;
    bool square_detected;

    bool c_direction_found;
    bool rotation_enabled = false;

    std::string victim_start = "";
    bool qr_flag = false;
    bool c_dir_flag = false;

    int averageAngle_1;
    int averageAngle_2;
    int averageAngle_3;
    int stabilize_count = 0;

    std::array<int, 9> angleFrequency = {0}; // -4 to 4
    int maxFrequency = 0;
    int maxFrequencyAngle = 0;
    int result_maxAngle = 0;
    int first_square_angle = 0;
    int last_square_angle = 0;
    bool square_direction_found = false;

    double prev_angle = 0;
    bool is_first = true;

    std_msgs::msg::Int32 msg1, msg2;

    bool square_tracked = false;
    cv::Point2f prev_square_center;
    cv::Point2f last_draw_center;

    void draw_and_track_square(
        const std::vector<cv::Point>& contour,
        const cv::Rect& roi_rect
    );

    bool circle_locked = false;
    cv::Point locked_center;
    int locked_radius = 0;
    bool log_once = true;
    const int STABILIZE_LIMIT = 200;
    int square_color = 0; // 0: 기본, 1: 흰색 바탕, 2: 검은색 바탕
    cv::Mat yolo_input;
    cv::Mat C_roi;
    bool first_c_latched = false;
    int  c_stable_count = 0;
    static constexpr int C_STABLE_TH = 5; // 5프레임 연속이면 안정

    cv::Point best_center{-1, -1};
    int best_radius = 0;
    int best_stable_count = 0;

    const int REQUIRED_STABLE = 5;
    bool c_locked = false;
    bool c_rotation_started = false;

    int c_rotation_state = 0; // 0: 회전X, 1: CW, 2: CCW
    BBoxData best_circle;     // 현재 사용 중인 안정적인 원


  private:
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_result;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_result_thermal;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_thermal;

    rclcpp::Subscription<custom_msgs::msg::BoundingBoxes>::SharedPtr subscription_;

    void imageCallBack(const sensor_msgs::msg::Image::SharedPtr msg_img);
    void imageCallBack_thermal(const sensor_msgs::msg::Image::SharedPtr msg_img);
    void bbox_callback(const custom_msgs::msg::BoundingBoxes::SharedPtr msg);

    void updateBoxes(std::vector<BBoxData> &slots, const std::vector<BBoxData> &new_boxes);

    std::vector<BBoxData> circle_c_;
    std::vector<BBoxData> circle_square_;
    std::vector<BBoxData> square_boxes;
  };

} // namespace vision_rescue_26