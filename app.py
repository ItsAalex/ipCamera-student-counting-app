import cv2
import numpy as np
import random
import os

# Load Yolo model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Hardcoded COCO classes
classes = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Generate colors for each class randomly
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Get the names of all the layers in the network
layer_names = net.getLayerNames()
# Get the indices of the output layers
try:
    output_layers_indices = net.getUnconnectedOutLayers().flatten()
except AttributeError:
    output_layers_indices = net.getUnconnectedOutLayers()

# Get the names of the output layers
output_layers = [layer_names[i - 1] for i in output_layers_indices]

ip = "16......" #Replace with your camera's IP address
username = "userexample"  # Replace with your username
password = "passwordexample"  # Replace with your password
address = f'rtsp://{username}:{password}@{ip}'
cap = cv2.VideoCapture(address)

# Check if camera opened successfully
if not cap.isOpened():
    print("Failed to open camera")
    exit()
else:
    print("Camera opened successfully")

# Create a named window
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

# Read camera frames until the user presses the 'q' key
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    # Retrieve the dimensions of the frame
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for output in outs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    font = cv2.FONT_HERSHEY_PLAIN
    person_count = 0
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            class_id = class_ids[i]
            if class_id < len(classes):
                label = str(classes[class_id])
                if label == "person":
                    person_count += 1
                    color = colors[class_id]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y + 30), font, 3, color, 2)

    # Add the number of counted people in the bottom right corner
    count_text = f"People Count: {person_count}"
    cv2.putText(frame, count_text, (width - 300, height - 20), font, 2, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()