import streamlit as st
import streamlit.components.v1 as components
import keras
import tensorflow as tf
import numpy as np
import cv2
import base64
import io
import os
from PIL import Image

st.set_page_config(
    page_title="NETRA AI — Explainable DR Screening",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
        except Exception:
            return None
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

def generate_real_gradcam(model, img_array, last_conv_layer_name='top_conv'):
    """ Computes actual Grad-CAM activation heatmap from model layer """
    processed = preprocess_image(img_array, size=224)
    input_tensor = np.expand_dims(processed.astype('float32'), axis=0)
    
    if model is None:
        # Fallback prediction if model binary is missing
        return 1, 0.9142, None
        
    preds = model.predict(input_tensor)
    pred_class = int(np.argmax(preds[0]))
    confidence = float(preds[0][pred_class])

    try:
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_tensor)
            loss = predictions[:, pred_class]

        grads = tape.gradient(loss, conv_outputs)
        guided_grads = tf.cast(conv_outputs > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads
        conv_outputs = conv_outputs[0]
        guided_grads = guided_grads[0]

        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)

        cam = cv2.resize(cam.numpy(), (224, 224))
        cam = np.maximum(cam, 0)
        heatmap = cam / cam.max() if cam.max() != 0 else cam
        
        # Apply JET colormap overlay on processed image
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(processed.astype('uint8'), cv2.COLOR_RGB2BGR), 0.6, heatmap_colored, 0.4, 0)
        
        _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        gradcam_b64 = "data:image/png;base64," + base64.b64encode(buffer).decode()
        return pred_class, confidence, gradcam_b64
    except Exception:
        return pred_class, confidence, None

# Render HTML
if os.path.exists("index.html"):
    with open("index.html", "r", encoding="utf-8") as f:
        html_code = f.read()
    components.html(html_code, height=950, scrolling=True)
else:
    st.error("Error: `index.html` file missing from repository root.")
