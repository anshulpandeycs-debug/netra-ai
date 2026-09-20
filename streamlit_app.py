import streamlit as st
import streamlit.components.v1 as components
import keras
import numpy as np
import cv2
import base64
import io
import os
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="NETRA AI - Explainable DR Screening",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Remove default Streamlit padding and headers
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

# Cache and Load Keras Model
@st.cache_resource
def load_netra_model():
    return keras.models.load_model("netraai_final.keras")

try:
    model = load_netra_model()
except Exception:
    model = None

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

GRADE_LABELS = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

# Read and Embed index.html safely
if os.path.exists("index.html"):
    with open("index.html", "r", encoding="utf-8") as f:
        html_code = f.read()

    component_value = components.html(html_code, height=1000, scrolling=True)

    # Process prediction request sent from JavaScript
    if component_value and isinstance(component_value, dict):
        if component_value.get("action") == "predict":
            image_b64 = component_value.get("image", "").split(",")[-1]
            img_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_array = np.array(img)

            processed = preprocess_image(img_array, size=224)
            input_array = np.expand_dims(processed.astype('float32'), axis=0)
            heatmap, pred_class, confidence = make_gradcam_heatmap(input_array, model)

            heatmap_resized = cv2.resize(heatmap, (224, 224))
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(cv2.cvtColor(processed.astype('uint8'), cv2.COLOR_RGB2BGR), 0.6, heatmap_colored, 0.4, 0)

            _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            overlay_b64 = "data:image/png;base64," + base64.b64encode(buffer).decode()

            st.components.v1.html(f"""
                <script>
                    window.parent.postMessage({{
                        type: "NETRA_PREDICTION",
                        grade: {pred_class},
                        gradeLabel: "{GRADE_LABELS[pred_class]}",
                        confidence: {confidence:.4f},
                        gradcam_image: "{overlay_b64}"
                    }}, "*");
                </script>
            """, height=0)
else:
    st.error("Error: `index.html` not found in the root directory.")
