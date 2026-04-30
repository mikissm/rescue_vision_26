# rescue_vision_26

## apriltag_ros 설치
```bash
cd ~/ros2_ws/src
git clone https://github.com/AprilRobotics/apriltag_ros.git
```

- 의존성 설치
```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```

- 빌드
```bash
colcon build --symlink-install
```

- 환경 설정
```bash
source install/setup.bash
```

- 에러 날 때
```bash
sudo apt install ros-humble-cv-bridge ros-humble-image-transport
sudo apt install ros-humble-apriltag
```

## 실행

- mapping

```bash
ros2 launch rescue_vision_26 mapping_launch.py
```

