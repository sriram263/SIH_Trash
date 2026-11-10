# from ultralytics import YOLO
# import cv2

# model = YOLO("best (1).pt")  # your trained weights

# cap = cv2.VideoCapture(0)  # open webcam

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(frame)  # run inference
#     annotated = results[0].plot()

#     cv2.imshow("YOLOv10 Detection", annotated)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

#Categories

# RECYCLABLE=['cardboard_box','can','plastic_bottle_cap','plastic_bottle','reuseable_paper']
# NON_RECYCLABLE=['plastic_bag','scrap_paper','stick','plastic_cup','snack_bag','plastic_box','straw','plastic_cup_lid','scrap_plastic','cardboard_bowl','plastic_cultery']
# HAZARDOUS=['battery','chemical_spray_can','chemical_plastic_bottle','chemical_plastic_gallon','light_bulb','paint_bucket']


# detect.py
from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("best (3).pt")   # change to your weights file
#best(3) is best so far from 0,1,2,3
# Put your phone's IP Webcam stream URL here
# Example: "http://192.168.1.5:8080/video"
ip_webcam_url = "http://192.168.1.17:8080/video"

# Open video stream from phone
cap = cv2.VideoCapture(ip_webcam_url)

if not cap.isOpened():
    print("❌ Could not open video stream. Check the IP and port.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        break

    # Run YOLO detection
    results = model(frame, imgsz=640, conf=0.5)

    # Annotate the frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("YOLOv10 Phone Camera", annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

