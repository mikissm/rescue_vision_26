# rescue_vision_26

# === apriltag_ros 설치 ===
cd ~/vision2601_ws/src
git clone https://github.com/AprilRobotics/apriltag_ros.git

# === 의존성 설치 ===
cd ~/vision2601_ws
rosdep install --from-paths src --ignore-src -r -y

# === 빌드 ===
colcon build --symlink-install

# === 환경 설정 ===
source install/setup.bash

# === 실행 (기본) ===
ros2 launch apriltag_ros apriltag.launch.py

# === Realsense 사용 시 ===
ros2 run apriltag_ros apriltag_node \
  --ros-args \
  -r image_rect:=/camera/camera/color/image_raw \
  -r camera_info:=/camera/camera/color/camera_info

# === 토픽 확인 ===
ros2 topic list
ros2 topic echo /tag_detections

# === 에러 날 때 ===
sudo apt install ros-humble-cv-bridge ros-humble-image-transport
sudo apt install ros-humble-apriltag

### 실행

- mapping

```bash
ros2 launch rescue_vision_26 mapping_launch.py
```

