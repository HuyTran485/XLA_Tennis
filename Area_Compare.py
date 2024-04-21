import cv2
import math
from ultralytics import YOLO
import threading
import time

Video_path = ('Test_data\Robot Is Like A Roomba For Tennis Balls.mp4')
Model_path = ('TennisBall.pt')
stop_event = threading.Event()


def tennisballDetection(Video_path, Model_path):
    model = YOLO(Model_path)
    classnames = ['Tennis Ball']

    cap = cv2.VideoCapture(Video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        max_area = 0  # Initialize maximum area
        largest_box_index = -1  # Initialize index of largest box

        frame_height, frame_width, _ = frame.shape
        camera_bottom_center = (frame_width // 2, frame_height)

        for i in range(1, 7):
            x_coordinate = frame_width // 7 * i
            cv2.line(frame, (x_coordinate, 0), (x_coordinate, frame_height), (255, 255, 255), 1, lineType=cv2.LINE_AA)

        result = model(frame, stream=True)
        result = list(result)
        for info in result:
            boxes = info.boxes
            for j, box in enumerate(boxes):
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                if confidence > 80:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    area = (x2 - x1) * (y2 - y1)  # Calculate area of bounding box
                    if area > max_area:
                        max_area = area
                        largest_box_index = j

                    cv2.putText(frame, f'Area: {area:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            for j, box in enumerate(boxes):
                if j == largest_box_index:
                    color = (0, 255, 0)  # Green color for the largest box
                else:
                    color = (0, 0, 255)
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                cv2.line(frame, (center_x, center_y), camera_bottom_center, color, 1, lineType=cv2.LINE_AA)

        num_divisions = 7
        division_width = frame_width // num_divisions

        if largest_box_index != -1:
            center_x, _ = result[0].boxes[largest_box_index].xyxy[0][0] + (
                        result[0].boxes[largest_box_index].xyxy[0][2] -
                        result[0].boxes[largest_box_index].xyxy[0][0]) / 2, \
                          result[0].boxes[largest_box_index].xyxy[0][1] + (
                                      result[0].boxes[largest_box_index].xyxy[0][3] -
                                      result[0].boxes[largest_box_index].xyxy[0][1]) / 2
            positions = ['Left 3', 'Left 2', 'Left 1', 'Forward', 'Right 1', 'Right 2', 'Right 3']
            ball_position_index = int(center_x // division_width)
            global position
            position = positions[min(ball_position_index, len(positions) - 1)]
            cv2.putText(frame, f'Ball Position: {position}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        2)
        else:
            position = 'None'
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def thread_two():
    while not stop_event.is_set():
        print(f"Ball Position: {position}")


if __name__ == "__main__":
    video_thread = threading.Thread(target=tennisballDetection, args=(Video_path, Model_path))
    video_thread.start()
    time.sleep(3)
    second_thread = threading.Thread(target=thread_two)
    second_thread.start()
    video_thread.join()
    stop_event.set()
    second_thread.join()
