# Eye Disease Detection using Streamlit

## 📌 Project Overview

This project is a **Streamlit-based web application** that uses **deep learning** to detect eye diseases from uploaded eye images. The application is designed to assist in **early screening** by providing fast and accessible predictions along with confidence scores.

The system leverages a **Convolutional Neural Network (CNN)** trained on eye images resized to **128×128 pixels**, making it suitable for image-based medical analysis.

---

## 🎯 Objectives

* Enable users to upload eye images through a simple web interface
* Perform automated eye disease detection using a trained CNN model
* Display predictions with confidence scores
* Prepare the application for cloud deployment using **Streamlit Community Cloud**

---

## 🧠 Model Description

* **Model Type:** Convolutional Neural Network (CNN)
* **Input Size:** 128 × 128 × 3
* **Framework:** TensorFlow / Keras
* **Task:** Image classification (cataract ,glaucoma , diabetic retinopathy, normal)

CNNs are used because they efficiently extract spatial features such as edges, textures, and abnormal patterns from images.

---

## ⚙️ Application Workflow

1. User uploads an eye image using the Streamlit interface
2. Image preprocessing is performed:

   * Resizing to 128×128
   * Normalization of pixel values
   * Reshaping to match model input format
3. The trained model performs inference on the processed image
4. The predicted class and confidence score are displayed instantly

---

## 🛠️ Technologies Used

* **Python**
* **Streamlit** – Web application framework
* **TensorFlow / Keras** – Deep learning model
* **NumPy** – Numerical operations
* **Pillow (PIL)** – Image processing
* **OpenCV** – Image handling (optional)

---

## 📂 Project Structure

```
Eye-Disease-Detection/
│
├── app.py                 # Streamlit application file
├── model/                 # Trained CNN model
│   └── eye_model.keras
├── requirements.txt       # Project dependencies
├── .gitignore             # Files to ignore in Git
├── README.md              # Project documentation
```

---

## 📦 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/hajar-hii/eye-disease-detection-streamlit.git
cd eye-disease-detection
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

---

## ☁️ Deployment

The application is intended to be deployed on **Streamlit Community Cloud**, enabling:

* Public access via a web URL
* Seamless deployment from GitHub
* No local setup for end users

---

## ⚠️ Limitations

* The model performance depends on the quality and diversity of training data
* This application is intended for **educational and screening purposes only**
* It does **not replace professional medical diagnosis**

---

## 🚀 Future Enhancements

* Support for multiple eye diseases
* Improved accuracy using larger datasets
* Mobile and telemedicine integration
* Doctor feedback and validation loop
* Multilingual user interface

---

## 👤 Author

Developed as an academic deep learning project for demonstrating the application of AI in healthcare screening.



## 📄 License

This project is for educational and research purposes.
