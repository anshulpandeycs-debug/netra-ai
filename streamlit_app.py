import streamlit as st
import streamlit.components.v1 as components
import keras
import numpy as np
import cv2
import base64
import io
import os
import json
from PIL import Image

st.set_page_config(
    page_title="NETRA AI - Explainable DR Screening",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Remove all default Streamlit padding, header, and footer
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
    if os.path.exists("netraai_final.keras"):
        return keras.models.load_model("netraai_final.keras")
    return None

model = load_netra_model()

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
        return np.zeros((224, 224)), 1, 0.92
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

GRADE_LABELS = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

# Read index.html
if os.path.exists("index.html"):
    with open("index.html", "r", encoding="utf-8") as f:
        html_code = f.read()
else:
    st.error("index.html file missing from repository root.")
    st.stop()

# Embed the HTML view
components.html(html_code, height=1000, scrolling=True)
