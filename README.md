# Echo Vision

**Echo Vision** is an intelligent assistive system that combines computer vision, voice interaction, and distance sensing to help users perceive and interpret their surroundings.  
It integrates **YOLOv8** for object detection, a **GPS module** for location tracking, and an **SR04 ultrasonic sensor** for distance measurement — all controlled through a Python-based AI system.

---

## 🚀 Features

- **Real-Time Object Detection**  
  Utilizes YOLOv8 for accurate and fast object identification using a connected camera.

- **Distance Calculation**  
  Employs the SR04 ultrasonic sensor to measure proximity between the user and detected objects.

- **Voice Interaction**  
  Integrates speech recognition and text-to-speech capabilities to provide spoken feedback and command control.

- **Location Awareness**  
  Uses a GPS module to identify and share the user’s location for navigation or safety purposes.

---

## 🧠 Tech Stack

- **Programming Language:** Python  
- **Libraries & Frameworks:**
  - `opencv-python` – Image capture and processing  
  - `speechrecognition` – Voice input processing  
  - `pvporcupine` – Wake-word detection  
  - `pyttsx3` – Speech synthesis for responses  
  - `ultrasonic-sensor` & `gps` – Hardware integration for real-world interaction  
  - `yolov8` – Deep learning model for object detection

---

## ⚙️ Hardware Components

- Raspberry Pi / Microcontroller (compatible with Python)
- USB / Pi Camera
- HC-SR04 Ultrasonic Sensor
- GPS Module
- Microphone & Speaker

---

## 🗂️ Project Structure

