import streamlit as st
import tensorflow as tf
import numpy as np
import tempfile
import pickle

# Load model and class indices once
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model("final_eye_cnn_from_scratch.keras")
    with open("class_indices.pkl", "rb") as f:
        class_indices = pickle.load(f)
    class_names = list(class_indices.keys())
    return model, class_names

model, class_names = load_model_and_classes()

def model_prediction(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(128,128))
    x = tf.keras.utils.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    idx = np.argmax(preds)
    return class_names[idx], preds[0][idx]

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Disease Identification", "About"])

# Home
if app_mode == "Home":
    st.title("Eye Disease Detection using CNN")
    st.markdown("""
    ### Supported Diseases
    - Normal Eye
    - Diabetic Retinopathy
    - Glaucoma
    - Cataract
    """)

# Disease Identification
elif app_mode == "Disease Identification":
    st.header("Upload Fundus Image")
    test_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(test_image.read())
            img_path = tmp.name

        st.image(test_image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            with st.spinner("Analyzing..."):
                label, confidence = model_prediction(img_path)

            st.success(f"Prediction: **{label}**")
            st.info(f"Confidence: **{confidence*100:.2f}%**")
            st.subheader("🩺 Disease Information")
            if label == "NORMAL":
                st.markdown("""
                **🟢 Normal Retina**

                **Description:**  
                The retina appears healthy with no visible signs of disease. The optic disc, blood vessels, and macula are within normal limits.

                **Recommendation:**  
                Maintain regular eye check-ups and follow healthy lifestyle practices to preserve vision.
                """)

            elif label == "DIABETIC_RETINOPATHY":
                st.markdown("""
                **🔵 Diabetic Retinopathy**

                **Description:**  
                A diabetes-related eye condition caused by damage to retinal blood vessels. Common findings include microaneurysms, hemorrhages, and fluid leakage.

                **Possible Symptoms:**  
                Blurred vision, dark spots, difficulty seeing at night.

                **Recommendation:**  
                Regular retinal screening and proper blood sugar control are essential. Consult an ophthalmologist for further evaluation.
                """)

            elif label == "GLAUCOMA":
                st.markdown("""
                **🟠 Glaucoma**

                **Description:**  
                A group of eye disorders that damage the optic nerve, often associated with increased intraocular pressure.

                **Possible Symptoms:**  
                Gradual loss of peripheral vision, often unnoticed in early stages.

                **Recommendation:**  
                Early detection is critical. Professional eye examination is strongly advised.
                """)

            elif label == "CATARACT":
                st.markdown("""
                **🟡 Cataract**

                **Description:**  
                A condition where the eye’s natural lens becomes cloudy, leading to reduced clarity of vision.

                **Possible Symptoms:**  
                Blurred vision, glare sensitivity, faded colors.

                **Recommendation:**  
                An eye specialist can assess severity and suggest appropriate treatment options.
                """)

            st.warning(
                "⚠ This information is for educational purposes only and does not replace professional medical diagnosis."
            )


# About
elif app_mode == "About":
    st.markdown("""
    ### CNN-based Eye Disease Detection
    This system uses a **custom CNN trained from scratch** to classify eye diseases
    from retinal images into four categories:
    - Normal
    - Diabetic Retinopathy
    - Glaucoma
    - Cataract


## 📊 About the Dataset

The dataset used in this project was obtained from **Kaggle**, contributed by **Gunaventhakodi**. It consists of labeled retinal fundus images collected for the purpose of automated eye disease classification using deep learning techniques.

The dataset contains high-quality color fundus images categorized into **four major eye conditions**:

* **Normal** – Healthy retinal images with no visible abnormalities
* **Diabetic Retinopathy** – Images showing microaneurysms, hemorrhages, and other diabetic-related retinal changes
* **Glaucoma** – Images indicating optic nerve damage and increased cup-to-disc ratio
* **Cataract** – Images exhibiting lens opacity and reduced retinal clarity

These categories represent some of the most common causes of vision impairment and blindness worldwide, making the dataset highly relevant for clinical screening and early diagnosis applications.

---

## 🗂 Dataset Organization

The dataset is organized into class-wise folders, with each folder containing retinal images corresponding to a specific eye condition. For model training and evaluation, the dataset was further split into **training, validation, and testing subsets** to ensure unbiased performance assessment.

All images were resized and normalized before training to maintain consistency and improve model convergence.

---

## 🎯 Purpose of Using This Dataset

This dataset was selected because:

* It provides **real-world retinal images** suitable for deep learning–based classification
* It supports **multi-class eye disease detection**, enabling the development of a single unified diagnostic model
* It is publicly available and well-suited for **academic research and learning purposes**

---

## ⚠ Disclaimer

This dataset is intended **only for research and educational use**.
The predictions generated by this system are **not a substitute for professional medical diagnosis** and should not be used for clinical decision-making without expert validation.

---

## 📌 Dataset Source

* **Platform:** Kaggle
* **Contributor:** Gunaventhakodi

* **Link:** [Kaggle Dataset](https://www.kaggle.com/datasets/gunaventhakodi/eye-disease-detection
""")
