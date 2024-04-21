import cv2
import math
from ultralytics import YOLO
import threading
import time

Video_path = ('Test_data/4 life-changing tennis ball retrievers.mp4')
#Video_path = (0)
Model_path = ('TennisBall.pt')
# Biến cờ để đồng bộ hóa giữa hai luồng
stop_event = threading.Event()


def tennisballDetection(Video_path, Model_path):
    # Initialize YOLO model
    model = YOLO(Model_path)

    # Reading the classes
    classnames = ['Tennis Ball']

    # Running real-time from webcam
    cap = cv2.VideoCapture(Video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Initialize variables to store information about the shortest line
        shortest_line_length = float('inf')
        shortest_line_index = -1

        # Initialize variable to store the camera bottom center
        frame_height, frame_width, _ = frame.shape
        camera_bottom_center = (frame_width // 2, frame_height)

        # Draw parallel lines to divide the screen into 7 equal parts
        for i in range(1, 7):
            x_coordinate = frame_width // 7 * i
            cv2.line(frame, (x_coordinate, 0), (x_coordinate, frame_height), (255, 255, 255), 1, lineType=cv2.LINE_AA)

        # Detect objects using YOLO
        result = model(frame, stream=True)
        result = list(result)  # Convert generator to list
        for info in result:
            boxes = info.boxes
            for j, box in enumerate(boxes):
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                if confidence > 80:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    # Calculate the center coordinates of the bbox
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    # Calculate the length of the line
                    line_length = math.sqrt((center_x - camera_bottom_center[0])**2 + (center_y - camera_bottom_center[1])**2)
                    # Check if the current line is shorter than the shortest line stored
                    if line_length < shortest_line_length:
                        shortest_line_length = line_length
                        shortest_line_index = j  # Save the index of the shortest line
                    # Write above the object
                    cv2.putText(frame, f'Length: {line_length:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Drawing lines with appropriate colors
            for j, box in enumerate(boxes):
                if j == shortest_line_index:
                    color = (0, 255, 0)  # Green color for the shortest line
                else:
                    color = (0, 0, 255)  # Red color for the remaining lines
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                # Draw Bounding Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                # Draw line from the center of the object to the midpoint of the bottom edge of the camera
                cv2.line(frame, (center_x, center_y), camera_bottom_center, color, 1, lineType=cv2.LINE_AA)

        # Calculate the position of the regions
        num_divisions = 7
        division_width = frame_width // num_divisions

        # Determine the position of the shortest line
        if shortest_line_index != -1:
            center_x, _ = result[0].boxes[shortest_line_index].xyxy[0][0] + (result[0].boxes[shortest_line_index].xyxy[0][2] - result[0].boxes[shortest_line_index].xyxy[0][0]) / 2, result[0].boxes[shortest_line_index].xyxy[0][1] + (result[0].boxes[shortest_line_index].xyxy[0][3] - result[0].boxes[shortest_line_index].xyxy[0][1]) / 2
            # Create a list of positions corresponding to the ordinal numbers from 0 to 6
            positions = ['Left 3', 'Left 2', 'Left 1', 'Forward', 'Right 1', 'Right 2', 'Right 3']
            # Determine the position of the ball
            ball_position_index = int(center_x // division_width)
            global position
            position = positions[min(ball_position_index, len(positions) - 1)]
            # Display the position of the ball on the screen
            cv2.putText(frame, f'Ball Position: {position}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            position = 'None'
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
        

    cap.release()
    cv2.destroyAllWindows()

def thread_two():
    while not stop_event.is_set():  # Check if the flag has not been set
        # Processing logic for thread 2 here
        print(f"Ball Position: {position}")


if __name__ == "__main__":
        # Create and start the processing video thread
    video_thread = threading.Thread(target=tennisballDetection, args=(Video_path, Model_path))
    video_thread.start()
    time.sleep(3)
    # Create and start the second thread
    second_thread = threading.Thread(target=thread_two)
    second_thread.start()
     # Wait for thread 1 to finish
    video_thread.join()
     # Set the flag to signal thread 2 that thread 1 has finished
    stop_event.set()
     # Wait for thread 2 to finish
    second_thread.join()
    
    
