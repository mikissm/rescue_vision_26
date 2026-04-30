from setuptools import setup

package_name = 'yolo_ros2_py'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=[
        'setuptools',
        'ultralytics',
        'opencv-python',
        'cv_bridge',
    ],
    zip_safe=True,
    maintainer='ssm',
    maintainer_email='mikissm06@gmail.com',
    description='ROS2 Python node for YOLOv8 detection',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'yolo_node = yolo_ros2_py.yolo_node:main',
        ],
    },
)
