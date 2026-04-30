from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    # 공통 파라미터
    camera_topic = "/camera/camera/color/image_raw"  # 모든 노드가 사용할 카메라 토픽

    # master 노드
    master_node = Node(
        package='rescue_vision_26',
        executable='master',
        name='master',
        parameters=[{'camera_topic': camera_topic}],
        output='screen'
    )

    # # hazmat 노드
    # hazmat_node = Node(
    #     package='rescue_vision_26',
    #     executable='hazmat',
    #     name='hazmat',
    #     parameters=[{'camera_topic': camera_topic}],
    #     output='screen'
    # )

    # # qr 노드
    # qr_node = Node(
    #     package='rescue_vision_26',
    #     executable='qr',
    #     name='qr',
    #     parameters=[{'camera_topic': camera_topic}],
    #     output='screen'
    # )

    # # findc 노드
    # findc_node = Node(
    #     package='rescue_vision_26',
    #     executable='findc',
    #     name='findc',
    #     parameters=[{'camera_topic': camera_topic}],
    #     output='screen'
    # )

    # vision_ui 노드
    vision_ui_node = Node(
        package='vision_ui',
        executable='vision_ui_node',
        name='vision_ui',
        parameters=[{'camera_topic': camera_topic}],
        output='screen'
    )

    return LaunchDescription([
        master_node,
        # hazmat_node,
        # qr_node,
        # findc_node,
        vision_ui_node
    ])