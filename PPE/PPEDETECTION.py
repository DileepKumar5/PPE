from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("ppe.pt")

# Define class names
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person',
              'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

# Only detect these six classes
allowedClasses = {'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Safety Vest'}

while True:
    success, img = cap.read()
    if not success:
        break  # Exit loop if no image is captured

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]  # Fixed the error

            # Process only the selected six classes
            if currentClass in allowedClasses and conf > 0.5:
                if currentClass in {'NO-Hardhat', 'NO-Safety Vest', 'NO-Mask'}:
                    myColor = (0, 0, 255)  # Red for missing safety gear
                else:
                    myColor = (0, 255, 0)  # Green for proper safety gear

                cvzone.putTextRect(img, f'{currentClass} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                   colorT=(255, 255, 255), colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit on 'q' key press

cap.release()
cv2.destroyAllWindows()
