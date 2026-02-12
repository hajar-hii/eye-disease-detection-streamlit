import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="EyeScan AI", layout="wide", page_icon="👁️")

# --- LOAD MODEL ---
@st.cache_resource
def load_my_model():
    # Ensure this filename matches exactly what you saved!
    return tf.keras.models.load_model("Trained_Model.keras")

model = load_my_model()
CLASS_NAMES = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

# --- GRAD-CAM FUNCTION (Explainable AI) ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # We create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradient of the output class with respect to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("👁️ Navigation")
page = st.sidebar.radio("Go to", ["Home", "Disease Identification", "Explainable AI (Grad-CAM)", "About"])

# --- HOME PAGE ---
if page == "Home":
    st.title("👁️ EyeScan AI: Advanced Diagnostics")
    st.write("Welcome to the AI-powered portal for detecting retinal diseases.")
    
    st.info("Our system uses the MobileNetV3Large architecture to provide 90% accurate diagnostic suggestions.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Why AI for Eye Care?
        Early detection of diseases like **Glaucoma** or **Diabetic Retinopathy** can prevent 80% of vision loss. 
        This tool helps bridge the gap where medical specialists may not be immediately available.
        """)
    with col2:
        st.markdown("""
        ### How to use:
        1. **Navigation:** Use the sidebar to go to the Identification page.
        2. **Upload:** Provide a clear retinal fundus image.
        3. **Explain:** Use the Grad-CAM page to see exactly what the AI detected.
        """)

# --- DISEASE IDENTIFICATION ---
elif page == "Disease Identification":
    st.title("🔍 Disease Identification")
    file = st.file_uploader("Upload Retinal Image (JPG/PNG)", type=["jpg", "png", "jpeg"])
    
    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Uploaded Image", width=300)
        
        # Preprocessing
        img_resized = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        
        if st.button("Run Diagnostic"):
            with st.spinner('Analyzing scan...'):
                predictions = model.predict(img_array)
                result = CLASS_NAMES[np.argmax(predictions)]
                confidence = np.max(predictions) * 100
                
                st.success(f"**Diagnosis:** {result}")
                st.write(f"**Confidence Score:** {confidence:.2f}%")
                st.progress(int(confidence))

# --- EXPLAINABLE AI ---
# --- EXPLAINABLE AI ---
elif page == "Explainable AI (Grad-CAM)":
    st.title("🧠 Explainable AI (Grad-CAM)")
    st.write("Grad-CAM highlights the specific areas of the eye the AI used to make its decision.")
    
    file = st.file_uploader("Upload for Heatmap Analysis", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file).convert("RGB")
        img_resized = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        
        # 1. Access the base MobileNet model
        try:
            base_model = model.get_layer('MobileNetV3Large')
            
            # 1. Look for a 4D layer safely
            last_conv_layer_name = None
            for layer in reversed(base_model.layers):
                try:
                    # Check if it has 4 dimensions (Batch, H, W, Channels)
                    if len(layer.output_shape) == 4:
                        last_conv_layer_name = layer.name
                        break
                except:
                    continue
            
            # 2. Fallback: If search failed, use the known MobileNetV3 feature layer
            if not last_conv_layer_name:
                # 'conv_1' is the standard final convolutional layer in MobileNetV3Large
                last_conv_layer_name = "conv_1" 

            if last_conv_layer_name:
                with st.spinner(f'Analyzing visual features from layer: {last_conv_layer_name}...'):
                    # --- GRAD-CAM MATH ---
                    grad_model = tf.keras.models.Model(
                        [base_model.inputs], 
                        [base_model.get_layer(last_conv_layer_name).output, base_model.output]
                    )
                    
                    # 1. Get the prediction from the ACTUAL model first
                    preds = model.predict(img_array)
                    class_idx = np.argmax(preds[0])

        # 2. Build the grad_model to look at the internal base_model
                    grad_model = tf.keras.models.Model([base_model.inputs], [base_model.get_layer(last_conv_layer_name).output, base_model.output])

        # 3. Use GradientTape to find the relationship between class and features
                    with tf.GradientTape() as tape:
                         last_conv_layer_output, base_preds = grad_model(img_array)
            # Use the class index we found from the main model
                         class_channel = base_preds[:, class_idx]

        # Calculate gradients
                    grads = tape.gradient(class_channel, last_conv_layer_output)
        
        # Check if grads are None (happens if layer isn't connected to output)
                    if grads is None:
                        st.error("Gradients could not be calculated. Try a different layer.")
                    else:
                        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
                    last_conv_layer_output = last_conv_layer_output[0]
                    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
                    heatmap = tf.squeeze(heatmap)
            
            # Normalize and display (same as before)
                    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
                    heatmap = heatmap.numpy()
                    # --- OVERLAY ---
                    heatmap_resized = cv2.resize(heatmap, (224, 224))
                    heatmap_resized = np.uint8(255 * heatmap_resized)
                    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                    
                    superimposed_img = heatmap_color * 0.4 + np.array(img_resized)
                    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

                    st.subheader("AI Decision Map")
                    st.image(superimposed_img, caption=f"Heatmap generated from layer: {last_conv_layer_name}", width=500)
                    st.success(f"The model identified patterns typical of **{CLASS_NAMES[class_idx]}** in the highlighted regions.")

        except Exception as e:
            st.error(f"Grad-CAM Error: {e}")
# --- ABOUT PAGE ---
elif page == "About":
    st.title("📝 About the Project")
    st.markdown("""
    ### Technical Specifications
    This project demonstrates **Transfer Learning** using a deep neural network optimized for clinical efficiency.
    """)
    
    st.table({
        "Attribute": ["Architecture", "Optimizer", "Input Resolution", "Parameters", "Accuracy"],
        "Details": ["MobileNetV3Large", "Adam (Exponential Decay)", "224x224 (RGB)", "Pre-trained ImageNet", "90% Test Accuracy"]
    })
    
    st.markdown("""
    ### Development Team Notes:
    * **Memory Optimization:** Implemented 8-unit batching to support 7.7GB RAM hardware.
    * **Validation Strategy:** 3-way split (Train/Val/Test) to ensure medical reliability.
    * **Transparency:** Integrated Grad-CAM to reduce the 'Black Box' nature of AI.
    """)
    st.warning("**Disclaimer:** This tool is for educational purposes only. It is NOT a substitute for professional medical diagnosis.")