# 🌊 Oil Spill Detection & Classification System

## 🚀 Overview

Oil spills pose a serious threat to marine ecosystems and require **early and accurate detection** for effective response. This project presents a **multi-stage hybrid Machine Learning and Deep Learning pipeline** for precise oil spill detection using **Synthetic Aperture Radar (SAR) images**.

The system combines **image processing, deep learning segmentation, and machine learning classification** to deliver highly accurate and reliable results in real-time through a web-based interface.

---

## 🎯 Key Features

* 🧠 Hybrid ML + DL architecture for improved accuracy
* 🛰️ Works on SAR satellite images (all-weather capability)
* 🔍 Pixel-level oil spill detection using segmentation models
* ⚡ Real-time prediction through web interface
* 📊 Reduces false positives using ML verification
* 🌐 User-friendly frontend for easy interaction

---

## 🏗️ System Architecture

The system follows a **multi-stage pipeline**:

1. **Preprocessing**

   * Lee Filter for noise reduction
   * Image normalization & enhancement

2. **Deep Learning Segmentation**

   * U-Net
   * LinkNet
   * DeepLabV3+
   * Ensemble approach for better accuracy

3. **Feature Extraction**

   * Texture (GLCM)
   * Intensity (mean, variance)
   * Shape (area, perimeter)

4. **Machine Learning Classification**

   * Random Forest
   * K-Nearest Neighbors (KNN)
   * Removes false detections

---

## 🧠 Technologies Used

### 💻 Programming & Frameworks

* Python
* TensorFlow / Keras
* OpenCV
* Scikit-learn

### 🌐 Web Technologies

* React.js (Frontend)
* Flask (Backend)

### 📊 Machine Learning

* Random Forest
* KNN

### 🤖 Deep Learning

* U-Net
* LinkNet
* DeepLabV3+

---

## 📂 Project Structure

```
TRAIL-4/
│── models/              # Trained ML & DL models
│── ml_dataset/          # Dataset
│── oil-spill-ui/        # Frontend (React)
│── backend/             # Flask backend
│── app.py               # Main backend file
│── requirements.txt
```

---

## ⚙️ Installation & Setup

### 🔹 1. Clone Repository

```
git clone https://github.com/ksathvikreddy31/oil_spill_detection.git
cd oil_spill_detection
```

### 🔹 2. Install Dependencies

```
pip install -r requirements.txt
```

### 🔹 3. Run Backend

```
python app.py
```

### 🔹 4. Run Frontend

```
cd oil-spill-ui
npm install
npm start
```

---

## 📊 Results

* Accurate detection of oil spill regions from SAR images
* Reduced false positives using ML verification layer
* Efficient performance with hybrid model approach
* Real-time visualization through web interface

---

## 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score

---

## 🌍 Applications

* Marine environmental monitoring
* Disaster response systems
* Oil spill tracking & prevention
* Remote sensing analytics

---

## ⚠️ Limitations

* Uses only SAR datasets (no multi-source data)
* Requires labeled data for training
* Limited generalization across different sensors

---

## 🔮 Future Enhancements

* Integration with real-time satellite APIs
* Mobile application support
* Transformer-based models for better accuracy
* Deployment on cloud platforms (AWS/Azure)

---

## 📄 Project Report

📥 Full Project Report available in this repository
(Refer to PDF for detailed methodology and results)

---

## 👨‍💻 Authors

* **Sathvik Reddy**
* **Raj Kumar**

---

## 🙏 Acknowledgement

Developed under the guidance of faculty at
**CMR College of Engineering & Technology**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
