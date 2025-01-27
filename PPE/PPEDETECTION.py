from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from datetime import datetime
from twilio.rest import Client  # Twilio API for WhatsApp
from dotenv import load_dotenv  # Load environment variables
import os  # Access OS environment variables

# Load environment variables from .env file
load_dotenv()

# Twilio Credentials (Securely loaded from .env)
account_sid = os.getenv("TWILIO_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER")
user_whatsapp_number = os.getenv("USER_WHATSAPP_NUMBER")
client = Client(account_sid, auth_token)

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("ppe.pt")

# Define class names
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person',
              'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

# Only detect these classes
allowedClasses = {'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Safety Vest'}

# Track alert timing
alert_start_time = None
previous_alert = ""

while True:
    success, img = cap.read()
    if not success:
        break  # Exit loop if no image is captured

    results = model(img, stream=True)

    alert_texts = []  # Store detected violations

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
            currentClass = classNames[cls]

            # Process only allowed classes
            if currentClass in allowedClasses and conf > 0.5:
                if currentClass in {'NO-Hardhat', 'NO-Safety Vest', 'NO-Mask'}:
                    myColor = (0, 0, 255)  # Red for missing safety gear
                    alert_texts.append(currentClass)  # Add violation to alert list
                else:
                    myColor = (0, 255, 0)  # Green for proper safety gear

                cvzone.putTextRect(img, f'{currentClass} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                   colorT=(255, 255, 255), colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    # **Sort detected violations to ensure consistent order**
    if alert_texts:
        alert_texts.sort()  # Sorting ensures "NO-Hardhat | NO-Mask" is always the same
        alert_message = " | ".join(alert_texts)  # Combine sorted detected issues

        cvzone.putTextRect(img, f"ALERT: {alert_message}",
                           (50, 50), scale=2, thickness=2, colorB=(0, 0, 255),
                           colorT=(255, 255, 255), colorR=(0, 0, 255), offset=10)

        # Track alert duration
        if alert_message == previous_alert:
            if alert_start_time is None:
                alert_start_time = time.time()  # Start timer for alert
            elif time.time() - alert_start_time >= 10:  # If the same alert persists for 10 seconds
                current_time = datetime.now().strftime("%H:%M:%S")
                try:
                    # Send WhatsApp Message via Twilio
                    message = client.messages.create(
                        from_=twilio_whatsapp_number,  # Twilio WhatsApp number
                        to=user_whatsapp_number,  # Your verified number
                        body=f"⚠️ Safety Alert: {alert_message} detected for 10 seconds at {current_time}!"
                    )
                    print("WhatsApp alert sent! Message SID:", message.sid)

                except Exception as e:
                    print("Error sending WhatsApp:", e)

                alert_start_time = None  # Reset alert timer after sending

        else:
            alert_start_time = None  # Reset timer if alert changes

        previous_alert = alert_message  # Update previous alert

    else:
        alert_start_time = None  # Reset if no violations

    cv2.imshow("PPE Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit on 'q' key press

cap.release()
cv2.destroyAllWindows()
