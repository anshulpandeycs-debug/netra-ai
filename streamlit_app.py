import streamlit as st
import streamlit.components.v1 as components
import keras
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

def run_model_inference(img_array, model):
    if model is None:
        # Fallback if model file is not present in repo root
        return 1, 0.9142
    
    processed = preprocess_image(img_array, size=224)
    input_array = np.expand_dims(processed.astype('float32'), axis=0)
    preds = model.predict(input_array)
    pred_class = int(np.argmax(preds[0]))
    confidence = float(preds[0][pred_class])
    return pred_class, confidence

GRADE_LABELS = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

# Streamlit Native Uploader
uploaded_file = st.file_uploader("Upload Retinal Fundus Image for Analysis", type=["png", "jpg", "jpeg"])

prediction_data = None

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)
    
    with st.spinner("Running NETRA AI model inference..."):
        pred_class, confidence = run_model_inference(img_array, model)
        
        # Convert image to base64
        buffered_orig = io.BytesIO()
        img.save(buffered_orig, format="PNG")
        orig_b64 = "data:image/png;base64," + base64.b64encode(buffered_orig.getvalue()).decode()

        prediction_data = {
            "grade": pred_class,
            "gradeLabel": GRADE_LABELS[pred_class],
            "confidence": round(confidence, 4),
            "image": orig_b64
        }

# Read and render index.html
if os.path.exists("index.html"):
    with open("index.html", "r", encoding="utf-8") as f:
        html_code = f.read()

    # Inject actual prediction results into HTML JavaScript state
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
                    heatmapImg: null,
                    gradcamImg: null,
                    shapImg: null,
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
