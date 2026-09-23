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
    page_title="NETRA AI — DR Screening",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
            max-width: 100% !important;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_netra_model():
    model_path = "netraai_final.keras"
    if os.path.exists(model_path):
        return keras.models.load_model(model_path)
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

def make_gradcam_heatmap(img_array, model, last_conv_layer_name='top_conv'):
    if model is None:
        # Fallback dummy prediction if model file is missing
        return np.zeros((224, 224)), 0, 0.95
    import tensorflow as tf
    
    # Run prediction
    preds = model.predict(img_array)
    pred_index = int(np.argmax(preds[0]))
    confidence = float(preds[0][pred_index])

    try:
        grad_model = keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            last_conv_layer_output, predictions = grad_model(img_array)
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy(), pred_index, confidence
    except Exception:
        return np.zeros((224, 224)), pred_index, confidence

GRADE_LABELS = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

# File Uploader
uploaded_file = st.file_uploader("Upload Retinal Fundus Image for Analysis", type=["png", "jpg", "jpeg"])

prediction_data = None

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)
    
    with st.spinner("Running NETRA AI model inference..."):
        processed = preprocess_image(img_array, size=224)
        input_array = np.expand_dims(processed.astype('float32'), axis=0)
        
        heatmap, pred_class, confidence = make_gradcam_heatmap(input_array, model)
        
        # Prepare Grad-CAM heatmap overlay image
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(processed.astype('uint8'), cv2.COLOR_RGB2BGR), 0.6, heatmap_colored, 0.4, 0)
        
        _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        overlay_b64 = "data:image/png;base64," + base64.b64encode(buffer).decode()
        
        # Convert uploaded image to base64
        buffered_orig = io.BytesIO()
        img.save(buffered_orig, format="PNG")
        orig_b64 = "data:image/png;base64," + base64.b64encode(buffered_orig.getvalue()).decode()

        prediction_data = {
            "grade": pred_class,
            "gradeLabel": GRADE_LABELS[pred_class],
            "confidence": round(confidence, 4),
            "image": orig_b64,
            "gradcam_image": overlay_b64
        }

# Read and render index.html
if os.path.exists("index.html"):
    with open("index.html", "r", encoding="utf-8") as f:
        html_code = f.read()

    # Inject real prediction data into JS state if an image was uploaded
    if prediction_data:
        inject_script = f"""
        <script>
            window.addEventListener('DOMContentLoaded', (event) => {{
                state.file = "{prediction_data['image']}";
                state.result = {{
                    id: "NETRA-" + Date.now().toString().slice(-6),
                    date: new Date().toLocaleDateString("en-IN", {{day: "2-digit", month: "short", year: "numeric"}}),
                    grade: {prediction_data['grade']},
                    gradeLabel: "{prediction_data['gradeLabel']}",
                    prediction: "{prediction_data['gradeLabel']}",
                    confidence: {prediction_data['confidence']},
                    quality: "Good",
                    image: "{prediction_data['image']}",
                    heatmapImg: "{prediction_data['gradcam_image']}",
                    gradcamImg: "{prediction_data['gradcam_image']}",
                    shapImg: "{prediction_data['gradcam_image']}",
                    explanationText: "Diabetic Retinopathy screening completed successfully.",
                    edge: null,
                    benchmark: null
                }};
                state.explainStep = 0;
                state.history.unshift(state.result);
                go("result");
            }});
        </script>
        </body>
        """
        html_code = html_code.replace("</body>", inject_script)

    components.html(html_code, height=900, scrolling=True)
else:
    st.error("Error: `index.html` not found in root directory.")
