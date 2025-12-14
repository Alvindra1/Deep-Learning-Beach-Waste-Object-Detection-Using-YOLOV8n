# Beach Waste Detection
# Using YOLOV8n 

This project is to make an AI to be able to detect waste in beaches. Utilizing YOLOV8n a deep learning model. The model will analyze the images to detect any kind of waste/trash scattered across the beach.

---

## 🌊 Project Overview

This application employs **YOLOv8n**, a lightweight yet effective deep-learning model, to analyze images of beaches and detect various forms of waste. By combining deep learning models with a user-friendly web interface.

The project includes a web application built with **Streamlit**, allowing users to upload images and view detection results.

---

## 🧠 Technologies Used

Here are the primary technologies and libraries utilized in this project:

* **YOLOv8n** – The nano version of the YOLOv8 model, chosen for its speed and efficiency.
* **Streamlit** – Used for creating an intuitive and interactive web application.
* **OpenCV** – A powerful computer vision library for image processing.
* **Ultralytics** – The official YOLO library used for training and inference.
* **PyTorch** – Deep learning framework underlying the model’s development and training.

---

## 📂 Dataset

The model was trained using the **BePLi Dataset v1: Beach Plastic Litter Dataset Version 1**, which provides a comprehensive collection of annotated beach images.

### Dataset Structure

* **Total Images:** 3,708
* **Train Images:** 2,226
* **Test Images:** 741
* **Validation Images:** 742

### Dataset Folders

* **Original Images**
* **Train Folder**
* **Test Folder**
* **Validation Folder**

### Dataset Link

The dataset is publicly available at:
[https://www.seanoe.org/data/00811/92297/](https://www.seanoe.org/data/00811/92297/)

---

## 🚀 Features

* Real-time object detection using YOLOv8n
* Web app interface for user-friendly interaction
* Detection bounding boxes and confidence scores

## How to run
1. Git clone repository
```
git clone https://github.com/Alvindra1/Deep-Learning-Beach-Waste-Object-Detection-Using-YOLOV8n
cd Deep-Learning-Beach-Waste-Object-Detection-Using-YOLOV8n
```
3. Create and activate a virtual environment
```
python -m venv venv
venv\Scripts\activate
```
5. Install dependancy
```
pip install -r requirements.txt
```
6.Run the Streamlit app
```
cd app
streamlit run streamlit_app.py
```
7. Open web browser and use the link that was given in the terminal by streamlit



