import os
import cv2
import face_recognition
import pandas as pd
from sklearn import neighbors
import joblib

# Load or Train the Face Recognition Model
model_filename = "face_recognition_model.clf"
knn_clf = None

if os.path.exists(model_filename):
    knn_clf = joblib.load(model_filename)
else:
    print("No trained model found. Please run train_model.py to train the model.")
    exit()

# Initialize Excel File for Attendance
attendance_file = "attendance.xlsx"
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Roll", "Name", "Timestamp"])
    df.to_excel(attendance_file, index=False)

# Initialize Camera
cam_port = 0
cam = cv2.VideoCapture(cam_port)

while True:
    ret, frame = cam.read()

    if not ret:
        print("Failed to capture frame. Exiting.")
        break

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding in face_encodings:
        predictions = knn_clf.predict([face_encoding])
        person_roll = predictions[0]

        # Capture attendance
        df = pd.read_excel(attendance_file)
        if person_roll not in df["Roll"].values:
            student_name = input(f"Enter student's name for roll {person_roll}: ")
            timestamp = pd.Timestamp.now()
            df = df.append({"Roll": person_roll, "Name": student_name, "Timestamp": timestamp}, ignore_index=True)
            df.to_excel(attendance_file, index=False)
            print(f"{student_name} marked attendance at {timestamp}")

        # Draw rectangle and text on the frame (similar to the original code)
        # ...

    # Display the frame and handle key events (similar to the original code)
    cv2.imshow("Smart Attendance", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
