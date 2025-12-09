# 🏃‍♂️ Lite-Action-Net: Real-Time Human Activity Recognition

A lightweight Computer Vision web application that detects human actions in real-time using a webcam. Built with **TensorFlow/Keras**, **OpenCV**, and deployed using **Streamlit**.

## 🚀 Project Overview
This project solves the problem of recognizing human behavior from video streams without requiring heavy hardware. 

Unlike traditional models that are 100s of MBs in size, this project uses a highly optimized **3MB custom Keras model**. It processes video frames live in the browser using `streamlit-webrtc`, making it fast and efficient for edge deployment.

## 🎯 Key Features
* **Real-Time Detection:** Processes live video feed instantly.
* **Lightweight Architecture:** The core model is only ~3MB.
* **7 Activity Classes:**
    * 👏 Clapping
    * 🤝 Meet and Split
    * 🪑 Sitting
    * 🧍 Standing Still
    * 🚶 Walking
    * 📖 Walking While Reading
    * 📱 Walking While Using Phone
* **Cloud Ready:** Fully optimized for deployment on Streamlit Cloud.

## 🛠️ Tech Stack
* **Language:** Python 3.9+
* **Deep Learning:** TensorFlow / Keras
* **Computer Vision:** OpenCV
* **Web Framework:** Streamlit & Streamlit-Webrtc

## 💻 How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
