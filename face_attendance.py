import cv2
import numpy as np
import pickle
import os
import csv
import time
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime


video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(r'I:\onr\new_onr\new_onr\haarcascade_frontalface_default.xml')


with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)

with open('data/face_data.pkl', 'rb') as f:
    FACES = pickle.load(f)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)


bg_path = r'I:\onr\new_onr\new_onr\kotak.jpg'
if os.path.exists(bg_path):
    image_background = cv2.imread(bg_path)
else:
    image_background = None  


col_names = ['NAME', 'TIME']
attendance_folder = "Attendance"
os.makedirs(attendance_folder, exist_ok=True)

print(" Attendance system started. Press 'q' to quit.")

while True:
    ret, frame = video.read()
    if not ret:
        print(" Could not read from camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop = frame[y:y + h, x:x + w, :]
        resize = cv2.resize(crop, (50, 50)).flatten().reshape(1, -1)

        output = knn.predict(resize)

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        filename = f"{attendance_folder}/attendance_{date}.csv"

        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 230), 2)
        cv2.putText(frame, output[0], (x, y - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        
        attendance = [str(output[0]), str(timestamp)]

        
        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(col_names)
            writer.writerow(attendance)

    
    if image_background is not None:
        
        bg_h, bg_w, _ = image_background.shape
        frame_resized = cv2.resize(frame, (min(640, bg_w), min(480, bg_h)))
        image_background[0:frame_resized.shape[0], 0:frame_resized.shape[1]] = frame_resized
        cv2.imshow('Attendance System', image_background)
    else:
        cv2.imshow('Attendance System', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()
print(" Attendance system stopped.")
