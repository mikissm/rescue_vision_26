from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    camera_topic = LaunchConfiguration("camera_topic")
    depth_topic = LaunchConfiguration("depth_topic")

    grayscale_node = Node(
        package="rescue_vision_26",
        executable="rgb_to_gray",
        name="rgb_to_gray",
        output="screen",
        parameters=[
            {
                "sub_topic": "/camera/camera/color/image_raw",
                "pub_topic": "/image_mono"
            }
        ]
    )


    apriltag_ros_node = Node(
        package="apriltag_ros",
        executable="apriltag_node",
        name="apriltag_node",
        output="screen",
        remappings=[
            ("image_rect", "/image_mono"),
            ("camera_info", "/camera/camera/color/camera_info"),
    ]
)

    hazmat_node = Node(
        package="rescue_vision_26",
        executable="hazmat",
        name="hazmat",
        output="screen",
        parameters=[
            {"img_width": 640},
            {"img_height": 360},
            {"camera_topic": camera_topic},
        ]
    )

    apriltag_node = Node(
        package="rescue_vision_26",
        executable="apriltag",
        name="apriltag",
        output="screen",
        parameters=[
            {"sub_topic": "/detections"},
            {"pub_topic": "/april/bounding_boxes"},
        ]
    )
    
    location_node = Node(
        package="rescue_vision_26",
        executable="location_calculation",
        name="location_calculation",
        output="screen",
        parameters=[
            {"color_topic": camera_topic},
            {"depth_topic": depth_topic},
            {"camera_info": "/camera/camera/color/camera_info"},

            {"show_window": True},
            {"pub_cloud": True},
            {"pub_marker": True},

            {"target_frame": "mani_base_link"},
            {"source_frame": "mani_camera_tf"},
        ]
    )

    return LaunchDescription([ 

        DeclareLaunchArgument(
            "camera_topic",
            default_value="/camera/camera/color/image_raw"
        ),

        DeclareLaunchArgument(
            "depth_topic",
            default_value="/camera/camera/aligned_depth_to_color/image_raw"
        ),

        grayscale_node,
        apriltag_ros_node,
        hazmat_node,
        apriltag_node,
        location_node
    ])
