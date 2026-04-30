cmake_minimum_required(VERSION 3.8)
project(rescue_vision_26)

# C++ 표준 설정
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(custom_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(apriltag_msgs REQUIRED)

# Include 디렉토리 설정
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

# 실행 파일 추가
add_executable(qr src/qr.cpp)
add_executable(hazmat src/hazmat.cpp)
add_executable(findc src/findc.cpp)
add_executable(master src/master.cpp)
add_executable(location_calculation src/location_calculation.cpp)
add_executable(apriltag src/apriltag)
add_executable(rgb_to_gray src/rgb_to_gray)

# 각 실행 파일의 의존성 설정
ament_target_dependencies(qr
  rclcpp
  std_msgs
  cv_bridge
  sensor_msgs
  image_transport
  custom_msgs
)

ament_target_dependencies(hazmat
  rclcpp
  std_msgs
  cv_bridge
  sensor_msgs
  image_transport
  custom_msgs
)

ament_target_dependencies(findc
  rclcpp
  std_msgs
  cv_bridge
  sensor_msgs
  image_transport
  custom_msgs
)

ament_target_dependencies(master
  rclcpp
  std_msgs
  cv_bridge
  sensor_msgs
  image_transport
  custom_msgs
)

ament_target_dependencies(location_calculation
  rclcpp
  std_msgs
  cv_bridge
  sensor_msgs
  image_transport
  custom_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  visualization_msgs
  tf2_eigen
)

ament_target_dependencies(apriltag
  rclcpp
  apriltag_msgs
  custom_msgs
)

ament_target_dependencies(rgb_to_gray
  rclcpp
  sensor_msgs
  cv_bridge
)

# OpenCV 라이브러리 링크
target_link_libraries(qr ${OpenCV_LIBS})
target_link_libraries(hazmat ${OpenCV_LIBS})
target_link_libraries(findc ${OpenCV_LIBS})
target_link_libraries(master ${OpenCV_LIBS})
target_link_libraries(location_calculation ${OpenCV_LIBS})
target_link_libraries(apriltag ${OpenCV_LIBS})

# 설치 규칙
install(TARGETS
  qr
  hazmat
  findc
  master
  location_calculation
  apriltag
  rgb_to_gray
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  yolo
  DESTINATION share/${PROJECT_NAME}
)
ament_package()
