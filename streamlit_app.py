import streamlit as st
import streamlit.components.v1 as components
import keras
import numpy as np
import cv2
import base64
import json
from PIL import Image
import io

# Page Configuration to match full-screen web app
st.set_page_config(
    page_title="NETRA AI — Explainable DR Screening", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Hide standard Streamlit header, footer, and padding via CSS
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
            padding-left: 0rem !important;
            padding-right: 0rem !important;
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

# Load Keras Model
@st.cache_resource
def load_netra_model():
    return keras.models.load_model("netraai_final.keras")

try:
    model = load_netra_model()
except Exception as e:
    model = None

# Model Helper Functions
def preprocess_image(img_array, size=224):
    img = cv2.resize(img_array, (size, size))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name='top_conv', pred_index=None):
    if model is None:
        return np.zeros((224, 224)), 0, 0.95
    import tensorflow as tf
    grad_model = keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    confidence = float(tf.nn.softmax(preds[0])[pred_index].numpy())
    return heatmap.numpy(), int(pred_index.numpy()), confidence

def generate_explanation(grade, heatmap):
    h, w = heatmap.shape
    quadrants = {
        'superior-nasal': heatmap[:h//2, :w//2].mean(), 
        'superior-temporal': heatmap[:h//2, w//2:].mean(),
        'inferior-nasal': heatmap[h//2:, :w//2].mean(), 
        'inferior-temporal': heatmap[h//2:, w//2:].mean(),
    }
    hot_region = max(quadrants, key=quadrants.get)
    findings = {
        0: "No visible diabetic retinopathy was detected. The blood vessels and retinal surface appear within normal limits.",
        1: "Mild non-proliferative diabetic retinopathy (NPDR) is present, with early microaneurysms identified.",
        2: "Moderate NPDR is present, with microaneurysms, hemorrhages, and early hard exudates visible.",
        3: "Severe NPDR is present, with extensive hemorrhages and significant microvascular changes.",
        4: "Proliferative diabetic retinopathy (PDR) is present, with abnormal new blood vessel growth detected.",
    }
    guidance = {
        0: "Continue routine annual screening and maintain good blood sugar and blood pressure control.",
        1: "Schedule a follow-up screening within 9-12 months and tighten glycemic control.",
        2: "Referral to an ophthalmologist is recommended within 3-6 months.",
        3: "Urgent referral to an ophthalmologist is recommended within weeks, not months.",
        4: "Immediate referral to a retina specialist is strongly recommended — this stage carries real risk of vision loss.",
    }
    return f"{findings[grade]} Attention was concentrated in the {hot_region} region.\n\n{guidance[grade]}"

GRADE_LABELS = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

# Read index.html content
try:
    with open("index.html", "r", encoding="utf-8") as f:
        html_code = f.read()
except FileNotFoundError:
    html_code = "<h1>Error: index.html not found in repository root directory.</h1>"

# Embed HTML in full-screen iframe component
components.html(html_code, height=1000, scrolling=True)
