#ifndef C_DETECTOR_NODE_HPP_
#define C_DETECTOR_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <custom_msgs/msg/bounding_boxes.hpp>

#include <opencv2/opencv.hpp>
#include <deque>
#include <map>

using namespace cv;

struct BBoxData
{
    std::string class_name;
    float confidence;
    float x1, y1, x2, y2;
    float cx, cy;
    bool valid = false;
};

class CDetectorNode : public rclcpp::Node
{
public:
    CDetectorNode();

private:
    /* ================= ROS ================= */
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_image_;
    rclcpp::Subscription<custom_msgs::msg::BoundingBoxes>::SharedPtr sub_bbox_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_debug_;

    rclcpp::TimerBase::SharedPtr timer_;

    /* ================= Params ================= */
    bool show_log = true;

    /* ================= Latest Data ================= */
    cv::Mat original;
    custom_msgs::msg::BoundingBoxes latest_bbox_;
    bool has_image_ = false;
    bool has_bbox_ = false;

    /* ================= Core ================= */
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void bbox_callback(const custom_msgs::msg::BoundingBoxes::SharedPtr msg);
    void timerCallback();

    void process();

    /* ================= 너 코드 ================= */
    void bbox_logic();
    void c_basic_setting();
    void circle_shape(int two_radius);
    void find_c_123(int two_radius);
    void detect_circle_rotation();

    /* ================= Utils ================= */
    bool check_black(const cv::Mat &img);
    bool safe_rect(const cv::Mat &img, const cv::Rect &r);

    /* ================= Data ================= */
    cv::Mat clone_mat, yolo_input, output_c, C_roi;

    std::vector<BBoxData> circle_c_;
    std::vector<cv::Vec3f> circles;

    cv::Vec3f c;
    bool find_two = false;

    /* 상태 */
    int c_rotation_state = 0;
    bool c_locked = false;

    int best_radius = 0;
    cv::Point best_center = cv::Point(-1, -1);

    std::vector<BBoxData> circle_square_;
    std::vector<BBoxData> square_boxes;

    int square_color = 0;

    bool c_rotation_started = false;
    bool first_c_latched = false;
    bool c_dir_flag = false;
    bool c_direction_found = false;
    bool rotation_enabled = false;

    int c_stable_count = 0;
    int best_stable_count = 0;
    int movement_count = 0;

    std::map<int,int> angleFrequency1;
    std::map<int,int> angleFrequency2;
    std::map<int,int> angleFrequency3;

    int maxFrequency1 = 0, maxFrequency2 = 0, maxFrequency3 = 0;
    int maxFrequencyAngle1 = 0, maxFrequencyAngle2 = 0, maxFrequencyAngle3 = 0;

    int result_maxAngle1 = 0, result_maxAngle2 = 0, result_maxAngle3 = 0;

    int averageAngle_1 = 0, averageAngle_2 = 0, averageAngle_3 = 0;

    const int REQUIRED_STABLE = 5;
    const int C_STABLE_TH = 5;
    const double PI = 3.141592;

    // OpenCV temp
    cv::Mat gray_clone;

    cv::Mat one_mat, two_mat, three_mat;
    cv::Mat one_gray, two_gray, three_gray;
    cv::Mat one_binary, two_binary, three_binary;

    // best circle
    BBoxData best_circle;

    // 함수
    void updateBoxes(std::vector<BBoxData>& target,
                    const std::vector<BBoxData>& input);

    // (선택) 히스토리
    std::deque<cv::Point2f> hist;
    cv::Point2f prev_center = cv::Point2f(-1, -1);
};

#endif