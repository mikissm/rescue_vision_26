import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np

# 커스텀 메시지 import
from custom_msgs.msg import BoundingBox, BoundingBoxes


class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node_py')

        # 카메라 토픽 고정
        self.declare_parameter('camera_topic', '/camera/camera/color/image_raw')
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value

        # 퍼블리셔 (이미지 + 바운딩박스)
        self.pub_image = self.create_publisher(Image, '/yolo/image', 10)
        self.pub_boxes = self.create_publisher(BoundingBoxes, '/yolo/bounding_boxes', 10)

        # 서브스크라이버
        self.sub = self.create_subscription(Image, self.camera_topic, self.callback, 10)

        self.bridge = CvBridge()

        # YOLOv8 모델 로드
        self.get_logger().info('Loading YOLOv8 model...')
        self.model = YOLO('/home/ssm/runs/detect/2026.01.16/weights/best.pt')
        self.get_logger().info('Model loaded!')

        # 클래스별 색상
        self.class_colors = {
            'circle_c': (0, 0, 255),        # 빨강
            'circle_square': (255, 0, 0),    # 파랑
            'circle_square_black': (0, 255, 0) # 초록
        }

        # 정확도 임계값
        self.conf_threshold = 0.5

        # OpenCV 창 설정
        cv2.namedWindow('YOLOv8 Realtime', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('YOLOv8 Realtime', 1280, 720)

    def callback(self, msg: Image):
        # ROS 이미지 → OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        print(frame.shape)

        # YOLO 추론
        results = self.model(frame)
        all_boxes_msg = BoundingBoxes()

        detections_by_class = {}  # {class_name: [(x1,y1,x2,y2,score,area), ...]}

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            names = result.names

            for box, score, class_id in zip(boxes, scores, class_ids):
                if score < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)
                class_name = names[class_id]

                if class_name not in detections_by_class:
                    detections_by_class[class_name] = []
                detections_by_class[class_name].append((x1, y1, x2, y2, float(score), area))

        # 사각형 관련 클래스 처리
        square_dets = []
        for cls in ['circle_square', 'circle_square_black']:
            if cls in detections_by_class:
                for d in detections_by_class[cls]:
                    square_dets.append((cls, *d))  # (class_name, x1,y1,x2,y2,score,area)

        # 면적 기준 내림차순 정렬
        square_dets = sorted(square_dets, key=lambda x: x[-1], reverse=True)

        # 전체 프레임 면적 기준 비율로 너무 큰 박스 거르기
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_w * frame_h
        max_area_ratio = 0.6  # 60% 이상 차지하는 박스는 무시

        filtered_squares = []
        for det in square_dets:
            area_ratio = det[-1] / frame_area
            if area_ratio < max_area_ratio:
                filtered_squares.append(det)

        # 너무 큰 것 제외 후 남은 것 중 면적 상위 1개 선택
        square_dets = sorted(filtered_squares, key=lambda x: x[-1], reverse=True)[:1]

        # circle_c 처리 (최대 2개, 중심 중첩 제거)
        selected_circles = []
        if 'circle_c' in detections_by_class:
            circle_dets = detections_by_class['circle_c']

            # 면적 내림차순 정렬
            circle_dets = sorted(circle_dets, key=lambda x: x[-1], reverse=True)

            for (x1, y1, x2, y2, score, area) in circle_dets:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # 기존 선택된 원과 중심 중첩 확인
                overlap = False
                for (_, sx1, sy1, sx2, sy2, _, _) in selected_circles:
                    if sx1 <= cx <= sx2 and sy1 <= cy <= sy2:
                        overlap = True
                        break

                if not overlap:
                    selected_circles.append(('circle_c', x1, y1, x2, y2, score, area))

                if len(selected_circles) >= 2:
                    break

        # 시각화 및 메시지 생성
        # 사각형
        for (class_name, x1, y1, x2, y2, score, area) in square_dets:
            label = f"{class_name} {int(score * 100)}%"
            color = (0, 255, 255) if class_name == "circle_square" else (255, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            box_msg = BoundingBox()
            box_msg.class_name = class_name
            box_msg.confidence = float(score)
            box_msg.x1 = float(x1)
            box_msg.y1 = float(y1)
            box_msg.x2 = float(x2)
            box_msg.y2 = float(y2)
            all_boxes_msg.boxes.append(box_msg)

        # circle_c
        for (class_name, x1, y1, x2, y2, score, area) in selected_circles:
            label = f"{class_name} {int(score * 100)}%"
            color = (0, 0, 255)  # 빨강

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            box_msg = BoundingBox()
            box_msg.class_name = class_name
            box_msg.confidence = float(score)
            box_msg.x1 = float(x1)
            box_msg.y1 = float(y1)
            box_msg.x2 = float(x2)
            box_msg.y2 = float(y2)
            all_boxes_msg.boxes.append(box_msg)

        # 퍼블리시
        self.pub_boxes.publish(all_boxes_msg)
        self.pub_image.publish(self.bridge.cv2_to_imgmsg(frame, 'bgr8'))

        # OpenCV 창 출력
        cv2.imshow('YOLOv8 Realtime', frame)
        cv2.waitKey(1)



def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
