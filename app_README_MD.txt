# 👁️ EyeScan AI: Comprehensive Project Documentation

## 🚀 1. Project Summary
This project is an end-to-end Deep Learning system designed to identify major eye diseases from retinal fundus images. By using a specialized neural network, the app provides instant diagnostic suggestions and visual proof of its reasoning.

**Key Goals:**
- Accurate classification of Cataracts, Glaucoma, and Diabetic Retinopathy.
- Transparency through Explainable AI (XAI).
- Seamless web-based deployment for clinical accessibility.

---

## 🧠 2. Technical Architecture (MobileNetV3Large)
The "brain" of this application is based on the **MobileNetV3Large** architecture. 
- **Efficiency:** Optimized for high performance even on systems with limited RAM (like 8GB setups).
- **Layers:** The model uses a series of Bottleneck blocks and Global Average Pooling.
- **Explainability:** We target the `conv_1` layer (the final 4D spatial layer) to extract feature maps for Grad-CAM analysis.

---

## 🧪 3. Explainable AI: Grad-CAM Logic
Instead of a "Black Box" approach, this app uses **Gradient-weighted Class Activation Mapping**. 
- **The Heatmap:** Red/Yellow regions represent areas where the AI found "high-impact" patterns (like vessel leaks or optic disc cupping).
- **Validation:** This allows doctors to verify that the AI is focusing on medical pathology rather than image noise or background lighting.

---

## 🛠️ 4. Local Execution & Setup
To run this project on your own machine:

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
2.Run the app
  streamlit run app2.py
3. Cloud Deployment: This project is configured for Streamlit Community Cloud. Ensure the Trained_Model.keras and requirements.txt are in the root directory before deploying.
5. File Manifest
app2.py: Main Streamlit application code.

Trained_Model.keras: The finalized weights for MobileNetV3.

requirements.txt: List of necessary Python libraries (TensorFlow, OpenCV, etc.).

Training_history.pkl: Recorded metrics from the model training phase.
