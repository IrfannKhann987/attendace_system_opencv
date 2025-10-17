# attendace_system_opencv
🧠 Attendance System using Face Detection (OpenCV)

This project is a simple yet practical implementation of Face Detection for automated attendance tracking.
It was built to solidify my understanding of OpenCV basics, which I learned from a 4-hour video tutorial.

📸 Overview

The system detects and recognizes faces in real time using OpenCV and logs attendance automatically.

Features

Captures 100 images per person and stores them in a dataset with their name.

Uses OpenCV’s face recognition to detect and identify individuals.

Logs attendance with name, date, and time in a .csv file.

Simple CLI-based interface for adding new faces and taking attendance.

🧩 Tech Stack

Python

OpenCV

NumPy

Pandas

⚙️ How It Works

Data Collection:
Run the script and enter your name. The system captures 100 images of your face and saves them in a dataset folder.

Training the Recognizer:
The recognizer learns from the stored images and maps each face to a unique label (person’s name).

Real-time Detection:
During attendance, the webcam feed runs face detection, matches recognized faces, and marks attendance.

Attendance Logging:
Each detection is stored with name, date, and time in a CSV file  ready to review anytime.

🚀 Future Improvements

Add GUI for user interaction

Store attendance in a database instead of CSV

Integrate email or WhatsApp alerts for attendance summaries
