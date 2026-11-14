# Gesture to Words 🤟

A deep learning-based **Indian Sign Language (ISL) gesture recognition system** developed as a collaborative **Mini Project**. The system recognizes sign language gestures from camera input and predicts the corresponding words.

## 📌 About the Project

**Gesture to Words** is an AI/ML-based project developed to recognize **Indian Sign Language gestures** using a trained deep learning model.

The system processes hand and body landmark sequences captured through a camera and predicts the corresponding ISL gesture in real time.

The project uses a combination of **publicly available Kaggle data** and **custom video data created by our team** for developing and training the gesture recognition system.

## ✨ Features

- Real-time Indian Sign Language gesture recognition
- Deep learning-based gesture classification
- Camera-based input
- Sequence-based gesture prediction
- Recognition of ISL gestures and conversion into corresponding words
- Confidence score for predictions
- Trained model included in the repository

## 🛠️ Technologies Used

- **Python**
- **TensorFlow / Keras**
- **MediaPipe**
- **OpenCV**
- **NumPy**
- **Google Colab**
- **Jupyter Notebook**

## 📂 Project Structure

```text
Gesture-to-Words/
│
├── code/
│   └── prediction_live.py
│
├── models/
│   ├── gesture_model.h5
│   └── classes.npy
│
├── notebooks/
│   └── trained_Colab.ipynb
│
├── test_camera.py
│
└── .gitignore
```

## 📊 Dataset

The dataset used for this project consists of data from two main sources:

### 1. Kaggle Dataset

A publicly available Indian Sign Language-related dataset obtained from **Kaggle** was used as one of the data sources for developing the gesture recognition model.

### 2. Custom Video Dataset

Additional video data was **recorded and created by our team** to supplement the available dataset and support the development of the gesture recognition system.

The video data was processed into suitable landmark sequences for model training and prediction.

The dataset will be published on **Kaggle** for reference and further academic use.

**Kaggle Dataset:** *Link will be added after publication.*

## 🧠 Model

The trained deep learning model is stored in:

```text
models/gesture_model.h5
```

The corresponding gesture class labels are stored in:

```text
models/classes.npy
```

The model is used by the live prediction script to classify the input gesture sequence and determine the corresponding word.

## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Shriraksha-Gourayya/Gesture-to-Words.git
cd Gesture-to-Words
```

### 2. Install the Required Libraries

```bash
pip install tensorflow mediapipe opencv-python numpy
```

### 3. Test the Camera

```bash
python test_camera.py
```

### 4. Run Live Gesture Recognition

```bash
python code/prediction_live.py
```

Make sure a working camera is connected and accessible to OpenCV.

## 🎯 Objective

The main objective of this project is to develop an AI-based system capable of recognizing **Indian Sign Language gestures** and converting them into understandable words.

The project aims to help reduce communication barriers between people who use sign language and people who may not be familiar with it.

## 👥 Team Members

This project was collaboratively developed by:

- **Ayush**
- **Amrutha K**
- **Yukthi NT**
- **Shriraksha Ravi Gourayya**

## 🎓 Project Type

**Mini Project**

## 🔮 Future Scope

- Expand the number of recognized ISL gestures.
- Improve recognition accuracy with a larger and more diverse dataset.
- Support continuous sign language sentence recognition.
- Develop a user-friendly application interface.
- Explore deployment on mobile and edge devices.
- Improve real-time performance and robustness under different lighting and background conditions.

## 📜 License

This project is developed for **academic and educational purposes**.
