from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='camera',
            namespace='camera',
            parameters=[
                {'camera_name': 'camera'},
                {'serial_no': '233522070302'},  # RealSense 시리얼 번호 #233522070244 #021222073502
                {'enable_color': True},
                {'enable_depth': True},
                {'enable_infra1': False},
                {'enable_infra2': False},
                {'enable_accel': False},
                {'enable_gyro': False},
                {'align_depth.enable': True}, 
                {'enable_sync': True},
                {'rgb_camera.color_profile': '640,360,30'},
                {'depth_module.depth_profile': '640,360,30'},
                {'rgb_camera.color_format': 'rgb8'},
                {'publish_tf': False},
            ]
        )
    ])
