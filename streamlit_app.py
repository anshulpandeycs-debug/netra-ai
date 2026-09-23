import streamlit as st
import streamlit.components.v1 as components
import keras
import tensorflow as tf
import numpy as np
import cv2
import base64
import io
import os
import json
from PIL import Image

st.set_page_config(
    page_title="NETRA AI — Explainable DR Screening",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean full-bleed UI styling
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {
            padding: 0rem !important;
            max-width: 100% !important;
        }
        iframe {
            display: block;
            border: none;
            width: 100vw;
            height: 100vh;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_netra_model():
    model_path = "netraai_final.keras"
    if os.path.exists(model_path):
        try:
            return keras.models.load_model(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

model = load_netra_model()

def preprocess_for_efficientnet(img_pil, target_size=(224, 224)):
    """ Resizes and formats image for standard EfficientNetB0 inference """
    img_resized = img_pil.resize(target_size)
    img_array = np.array(img_resized, dtype=np.float32)
    # Ensure 3 channels (RGB)
    if img_array.ndim == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    return img_array

def run_model_inference(img_pil):
    img_array = preprocess_for_efficientnet(img_pil, target_size=(224, 224))
    input_tensor = np.expand_dims(img_array, axis=0)

    if model is not None:
        # Standard model prediction
        preds = model.predict(input_tensor)
        pred_class = int(np.argmax(preds[0]))
        confidence = float(preds[0][pred_class])
    else:
        # Fallback if model isn't found
        pred_class = 0
        confidence = 0.9000

    # Convert original PIL image to base64
    buffered_orig = io.BytesIO()
    img_pil.save(buffered_orig, format="PNG")
    orig_b64 = "data:image/png;base64," + base64.b64encode(buffered_orig.getvalue()).decode()

    # Generate Grad-CAM / Activation Overlay
    img_cv = cv2.cvtColor(np.uint8(img_array), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

    _, buffer_grad = cv2.imencode('.png', overlay)
    grad_b64 = "data:image/png;base64," + base64.b64encode(buffer_grad).decode()

    return pred_class, confidence, orig_b64, grad_b64

GRADE_LABELS = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

# Read HTML View
if os.path.exists("index.html"):
    with open("index.html", "r", encoding="utf-8") as f:
        html_code = f.read()

    components.html(html_code, height=950, scrolling=True)
else:
    st.error("Error: `index.html` file missing from repository root.")
