#include "rescue_vision_26/location_calculation.hpp"

LocationCalculation::LocationCalculation()
: Node("yolo_plane_mapper")
{
    // ------------------ 기본 파라미터 ------------------
    declare_parameter("color_topic", "/camera/camera/color/image_raw");
    declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw");
    declare_parameter("camera_info", "/camera/camera/color/camera_info");

    // ------------------ 시각화 토글 파라미터 ------------------
    declare_parameter("show_window", true);   // OpenCV 창 띄울지
    declare_parameter("pub_cloud", true);     // PointCloud 퍼블리시 여부
    declare_parameter("pub_marker", true);     // MarkerArray 퍼블리시 여부

    // ------------------ tf 파라미터 ------------------
    declare_parameter("target_frame", "mani_base_link");
    declare_parameter("source_frame", "mani_camera_tf");

    // ------------------ 파라미터 로드 ------------------
    show_window_ = get_parameter("show_window").as_bool();
    pub_cloud_   = get_parameter("pub_cloud").as_bool();
    pub_marker_  = get_parameter("pub_marker").as_bool();

    auto color_topic = get_parameter("color_topic").as_string();
    auto depth_topic = get_parameter("depth_topic").as_string();
    auto camera_info = get_parameter("camera_info").as_string();

    target_frame_ = get_parameter("target_frame").as_string();
    source_frame_ = get_parameter("source_frame").as_string();

    // ------------------ TF2 ------------------
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // ------------------ 구독 ------------------
    color_sub_ = create_subscription<sensor_msgs::msg::Image>(
        color_topic, 10,
        std::bind(&LocationCalculation::colorCallback, this, std::placeholders::_1));

    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
        depth_topic, 10,
        std::bind(&LocationCalculation::depthCallback, this, std::placeholders::_1));

    boxes_sub_ = create_subscription<custom_msgs::msg::BoundingBoxes>(
        "/sign/bounding_boxes", 10,
        std::bind(&LocationCalculation::boxesCallback, this, std::placeholders::_1));

    april_boxes_sub_ = create_subscription<custom_msgs::msg::BoundingBoxes>(
        "/april/bounding_boxes", 10,
        std::bind(&LocationCalculation::boxesAprilCallback, this, std::placeholders::_1));

    caminfo_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info, 10,
        std::bind(&LocationCalculation::cameraInfoCallback, this, std::placeholders::_1));

    // ------------------ 퍼블리셔 ------------------
    pub_location_ = this->create_publisher<custom_msgs::msg::SignData>(
        "/object_location", 10);

    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/object_markers", 10);

    // ------------------ 타이머 ------------------
    timer_ = create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&LocationCalculation::timerCallback, this));

    // ------------------ 상태 출력 ------------------
    RCLCPP_INFO(get_logger(), "FaceEquation ready");
    RCLCPP_INFO(get_logger(), "  show_window: %s", show_window_ ? "true" : "false");
    RCLCPP_INFO(get_logger(), "  pub_cloud:   %s", pub_cloud_   ? "true" : "false");
    RCLCPP_INFO(get_logger(), "  pub_marker:  %s", pub_marker_  ? "true" : "false");
}

void LocationCalculation::colorCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    try {
        auto cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        std::lock_guard<std::mutex> lock(data_mutex_);
        color_img_ = cv_ptr->image.clone();
        last_color_stamp_ = msg->header.stamp;
    } catch (...) {}
}

void LocationCalculation::depthCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    try {
        auto cv_ptr = cv_bridge::toCvCopy(msg, "16UC1");
        std::lock_guard<std::mutex> lock(data_mutex_);
        depth_img_ = cv_ptr->image.clone();
    } catch (...) {}
}

void LocationCalculation::boxesCallback(
    const custom_msgs::msg::BoundingBoxes::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(data_mutex_);

    std::vector<Object> new_objects;

    for (auto &bb : msg->boxes)
    {
        Object obj;

        obj.object_name = bb.class_name;

        obj.roi = cv::Rect(
            bb.x1, bb.y1,
            bb.x2 - bb.x1,
            bb.y2 - bb.y1
        );

        obj.id = bb.confidence;
        obj.center_cam = cv::Point3f(0, 0, 0);
        obj.center_base = cv::Point3f(0, 0, 0);
        obj.detect_count = 0;
        obj.lost_count = 0;
        obj.confirmed = false;
        obj.active = true;
        obj.last_update = this->now();

        new_objects.push_back(obj);
    }

    objects_sign_ = new_objects;
}

void LocationCalculation::boxesAprilCallback(
    const custom_msgs::msg::BoundingBoxes::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(data_mutex_);

    std::vector<Object> new_objects;

    for (auto &bb : msg->boxes)
    {
        Object obj;

        obj.object_name = bb.class_name;

        obj.roi = cv::Rect(
            bb.x1, bb.y1,
            bb.x2 - bb.x1,
            bb.y2 - bb.y1
        );

        obj.id = bb.confidence;
        obj.center_cam = cv::Point3f(0, 0, 0);
        obj.center_base = cv::Point3f(0, 0, 0);
        obj.detect_count = 0;
        obj.lost_count = 0;
        obj.confirmed = false;
        obj.active = true;
        obj.last_update = this->now();

        new_objects.push_back(obj);
    }

    objects_april_ = new_objects;
}

void LocationCalculation::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(data_mutex_);
    fx_ = msg->k[0];
    fy_ = msg->k[4];
    cx_ = msg->k[2];
    cy_ = msg->k[5];
    // RCLCPP_INFO_THROTTLE(
    //     get_logger(), *get_clock(), 5000,
    //     "Camera intrinsics updated: fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
    //     fx_, fy_, cx_, cy_);
}

void LocationCalculation::timerCallback()
{
    std::lock_guard<std::mutex> lock(data_mutex_);

    // 합치기
    objects_.clear();

    // april 추가
    objects_.insert(objects_.end(),
                objects_april_.begin(), objects_april_.end());

    // hazmat 필터링 후 추가
    for (const auto& sign : objects_sign_)
    {
        bool overlap = false;

        for (const auto& april : objects_april_)
        {
            if ((sign.roi & april.roi).area() > 0)  // 겹침 체크
            {
                overlap = true;
                break;
            }
        }

        if (!overlap)
        {
            objects_.push_back(sign);
        }
    }

    // 처리
    for (auto &obj : objects_)
    {
        obj.center_cam = getCenterXYZFromROI(obj.roi);

        obj.center_base = transformToFrame(
            obj.center_cam,
            target_frame_,
            source_frame_);

        publish_data(obj, obj.center_cam);
    }

    draw_objects(objects_);
    publish_markers(objects_);

    // 다음 프레임 대비 초기화
    objects_.clear();
}

cv::Point3f LocationCalculation::getCenterXYZFromROI(const cv::Rect& roi)
{
    cv::Point center(roi.x + roi.width / 2,
                     roi.y + roi.height / 2);

    int half = 3;

    float sum = 0.0f;
    int count = 0;

    for (int dy = -half; dy <= half; dy++)
    {
        for (int dx = -half; dx <= half; dx++)
        {
            int x = center.x + dx;
            int y = center.y + dy;

            if (x < 0 || x >= depth_img_.cols || y < 0 || y >= depth_img_.rows)
                continue;

            uint16_t d = depth_img_.at<uint16_t>(y, x);
            if (d == 0) continue;

            sum += d;
            count++;
        }
    }

    if (count < 5)
    {
        return cv::Point3f(0, 0, 0); // 실패 신호
    }

    float d_avg = sum / count;

    float Z = d_avg * 0.001f;
    float X = (center.x - cx_) * Z / fx_;
    float Y = (center.y - cy_) * Z / fy_;

    return cv::Point3f(X, Y, Z);
}

cv::Point3f LocationCalculation::transformToFrame(
    const cv::Point3f &pt_in,
    const std::string &target_frame,
    const std::string &source_frame)
{
    try
    {
        geometry_msgs::msg::TransformStamped tf =
            tf_buffer_->lookupTransform(
                target_frame, source_frame, tf2::TimePointZero);

        Eigen::Affine3d tf_eigen = tf2::transformToEigen(tf);

        Eigen::Vector3d p(pt_in.x, pt_in.y, pt_in.z);
        Eigen::Vector3d p_out = tf_eigen * p;

        return cv::Point3f(
            p_out.x(),
            p_out.y(),
            p_out.z());
    }
    catch (tf2::TransformException &ex)
    {
        RCLCPP_WARN(this->get_logger(),
            "TF error (%s -> %s): %s",
            source_frame.c_str(),
            target_frame.c_str(),
            ex.what());

        return pt_in;
    }
}

void LocationCalculation::publish_data(
    const Object &obj,
    const cv::Point3f &location)
{
    custom_msgs::msg::SignData msg;

    msg.name = obj.object_name;
    msg.id = obj.id;

    msg.position.x = location.x;
    msg.position.y = location.y;
    msg.position.z = location.z;

    pub_location_->publish(msg);
}

void LocationCalculation::draw_objects(const std::vector<Object> &objects)
{
    if (!show_window_ || color_img_.empty())
        return;

    cv::Mat vis = color_img_.clone();

    for (const auto &obj : objects)
    {
        // bbox
        cv::rectangle(vis, obj.roi, cv::Scalar(0, 255, 0), 2);

        // label
        std::string label = obj.object_name + " ID:" + std::to_string(obj.id);

        cv::putText(vis, label,
            cv::Point(obj.roi.x, obj.roi.y - 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.5,
            cv::Scalar(0, 255, 0), 1);

        // 좌표
        char coord[100];
        snprintf(coord, sizeof(coord),
            "(%.2f, %.2f, %.2f)",
            obj.center_cam.x,
            obj.center_cam.y,
            obj.center_cam.z);

        cv::putText(vis, coord,
            cv::Point(obj.roi.x, obj.roi.y + obj.roi.height + 15),
            cv::FONT_HERSHEY_SIMPLEX, 0.4,
            cv::Scalar(0, 255, 0), 1);
    }

    cv::imshow("Detection", vis);
    cv::waitKey(1);
}

void LocationCalculation::publish_markers(const std::vector<Object> &objects)
{
    if (!pub_marker_)
        return;

    visualization_msgs::msg::MarkerArray marker_array;

    int marker_id = 0;

    for (const auto &obj : objects)
    {
        if (obj.center_cam.z == 0) continue; // invalid skip

        visualization_msgs::msg::Marker marker;

        marker.header.frame_id = "mani_base_link";
        marker.header.stamp = this->now();

        marker.ns = "objects";
        marker.id = marker_id++;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;

        // 위치
        marker.pose.position.x = obj.center_cam.x;
        marker.pose.position.y = obj.center_cam.y;
        marker.pose.position.z = obj.center_cam.z;

        marker.pose.orientation.w = 1.0;

        // 크기
        marker.scale.x = 0.05;
        marker.scale.y = 0.05;
        marker.scale.z = 0.05;

        // 색상 (ID 기반)
        cv::Scalar color = base_colors_[obj.id % base_colors_.size()];

        marker.color.r = color[2] / 255.0;
        marker.color.g = color[1] / 255.0;
        marker.color.b = color[0] / 255.0;
        marker.color.a = 1.0;

        marker.lifetime = rclcpp::Duration::from_seconds(0.3);

        marker_array.markers.push_back(marker);

        // 텍스트 marker 추가
        visualization_msgs::msg::Marker text_marker;

        text_marker.header = marker.header;
        text_marker.ns = "labels";
        text_marker.id = marker_id++;
        text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::msg::Marker::ADD;

        text_marker.pose.position = marker.pose.position;
        text_marker.pose.position.z += 0.1;

        text_marker.scale.z = 0.05;

        text_marker.color.r = 1.0;
        text_marker.color.g = 1.0;
        text_marker.color.b = 1.0;
        text_marker.color.a = 1.0;

        text_marker.text = obj.object_name + " ID:" + std::to_string(obj.id);

        text_marker.lifetime = rclcpp::Duration::from_seconds(0.3);

        marker_array.markers.push_back(text_marker);
    }

    marker_pub_->publish(marker_array);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LocationCalculation>());
    rclcpp::shutdown();
    return 0;
}