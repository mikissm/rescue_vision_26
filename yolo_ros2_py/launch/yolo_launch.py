from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='yolo_ros2_py',
            executable='yolo_node',
            name='yolo_node_py',
            output='screen'
        )
    ])
