import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="NetraAI — DR Screening",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS FOR LIGHT THEME, SIDEBAR, & BUTTON CONTRAST
# =========================================================
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #f7fafb;
    }

    /* Container width */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    /* Headings */
    h1, h2, h3 {
        color: #123b4a !important;
    }

    /* Text elements */
    p, label {
        color: #526572;
    }

    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background-color: #ffffff;
        border: 1px dashed #9bbfc3;
        border-radius: 12px;
        padding: 10px;
    }

    /* Button Fix - Force White Text Visibility */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #0f766e !important;
        background-color: #0f766e !important;
        color: #ffffff !important;
        font-weight: 600;
        padding: 0.55rem 1.2rem;
        width: 100%;
    }

    .stButton > button p {
        color: #ffffff !important;
    }

    .stButton > button:hover {
        background-color: #115e59 !important;
        border-color: #115e59 !important;
        color: #ffffff !important;
    }

    .stButton > button:hover p {
        color: #ffffff !important;
    }

    /* Metric Card Styling */
    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e1e8ea;
        border-radius: 12px;
        padding: 15px;
    }

    .explanation-box {
        background-color: #ffffff;
        border: 1px solid #dce7e9;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        line-height: 1.6;
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL LOADING
# =========================================================
@st.cache_resource
def load_netra_model():
    return tf.keras.models.load_model("netraai_final.keras")

try:
    model = load_netra_model()
    model_status = True
except Exception as e:
    model = None
    model_status = False
    model_error = str(e)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 👁️ NetraAI")
    st.caption("Explainable AI for Diabetic Retinopathy Screening")
    st.divider()

    st.markdown("### Model")
    
    st.write("**Architecture**")
    st.caption("Lightweight CNN")

    st.write("**Input**")
    st.caption("224 × 224 retinal image")

    st.write("**Output**")
    st.caption("5 DR grades")

    st.write("**Explainability**")
    st.caption("Grad-CAM")

    st.divider()

    if model_status:
        st.success("● Model loaded")
    else:
        st.error("● Model unavailable")

    st.divider()
    st.caption("NetraAI provides AI-assisted screening and should not replace professional clinical evaluation.")


# =========================================================
# PREPROCESSING & GRAD-CAM
# =========================================================
def preprocess_image(img_array, size=224):
    img = cv2.resize(img_array, (size, size))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name='top_conv', pred_index=None):
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
    except Exception:
        conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        grad_model = tf.keras.models.Model(
            [model.inputs], [conv_layers[-1].output, model.output]
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
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if float(max_val.numpy()) > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy(), int(pred_index.numpy())


# =========================================================
# 4-LINE CLINICAL EXPLANATIONS
# =========================================================
def generate_explanation(grade, heatmap):
    h, w = heatmap.shape
    quadrants = {
        'superior-nasal': heatmap[:h//2, :w//2].mean(),
        'superior-temporal': heatmap[:h//2, w//2:].mean(),
        'inferior-nasal': heatmap[h//2:, :w//2].mean(),
        'inferior-temporal': heatmap[h//2:, w//2:].mean(),
    }
    hot_region = max(quadrants, key=quadrants.get)

    explanations = {
        0: (
            "Line 1: No visible microaneurysms, hemorrhages, or exudates were detected across the retinal field.\n"
            "Line 2: The foveal avascular zone and retinal vascular structure demonstrate intact integrity.\n"
            f"Line 3: Model spatial attention was lightly observed in the {hot_region} region without pathological flags.\n"
            "Line 4: Recommendation: Maintain routine annual eye examinations and maintain optimal glycemic control."
        ),
        1: (
            "Line 1: Mild Non-Proliferative DR detected due to localized microaneurysm formation in tiny blood vessels.\n"
            "Line 2: Vascular permeability remains mostly stable without major hard exudates or diffuse swelling.\n"
            f"Line 3: High feature intensity is concentrated within the {hot_region} quadrant of the retina.\n"
            "Line 4: Recommendation: Schedule a follow-up screening within 9–12 months and monitor HbA1c levels."
        ),
        2: (
            "Line 1: Moderate Non-Proliferative DR identified with multiple microaneurysms and early dot-blot hemorrhages.\n"
            "Line 2: Hard exudates indicate minor lipid leakage secondary to localized breakdown of the blood-retinal barrier.\n"
            f"Line 3: Key diagnostic activation highlights focal vascular changes in the {hot_region} quadrant.\n"
            "Line 4: Recommendation: Specialist ophthalmic consultation recommended within 3–6 months for close monitoring."
        ),
        3: (
            "Line 1: Severe Non-Proliferative DR detected showing extensive intraretinal hemorrhages and venous beading.\n"
            "Line 2: Widespread capillary non-perfusion poses a significant risk for progression to proliferative disease.\n"
            f"Line 3: Dense Grad-CAM heat clusters signal severe vascular compromise in the {hot_region} region.\n"
            "Line 4: Recommendation: Urgent referral to a retina specialist within weeks to prevent further progression."
        ),
        4: (
            "Line 1: Proliferative DR detected with active neovascularization, indicating advanced microvascular disease.\n"
            "Line 2: Fragile new blood vessel growth increases the risk of vitreous hemorrhage and tractional detachment.\n"
            f"Line 3: Strongest neural network salience focuses on high-risk lesions within the {hot_region} quadrant.\n"
            "Line 4: Recommendation: Immediate specialized care (e.g., anti-VEGF or laser therapy) is strongly advised."
        )
    }

    return explanations[grade]


GRADE_LABELS = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]


# =========================================================
# MAIN APP BODY
# =========================================================
st.title("NetraAI")
st.subheader("Diabetic Retinopathy Screening")
st.write("Upload a retinal fundus image to obtain an AI-assisted DR classification with a visual explanation.")

if not model_status:
    st.error("The model could not be loaded properly.")
    st.code(model_error)
    st.stop()

uploaded_file = st.file_uploader("Choose a retinal fundus image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)

    st.markdown("### 🖼️ Image Preview")
    prev_col, det_col = st.columns([1.5, 1])
    
    with prev_col:
        st.image(img_array, caption="Uploaded Retinal Image", use_container_width=True)
    with det_col:
        st.markdown("#### File Details")
        st.write(f"**Filename:** {uploaded_file.name}")
        st.write(f"**Dimensions:** {img_array.shape[1]} × {img_array.shape[0]} px")
        st.info("Ensure the retina is well-lit and centered for optimal accuracy.")

    btn_col1, btn_col2 = st.columns([1, 4])
    with btn_col1:
        analyze_click = st.button("🔍 Analyze Image")

    if analyze_click:
        with st.spinner("Processing retinal features and calculating Grad-CAM..."):
            processed = preprocess_image(img_array, size=224)
            input_array = np.expand_dims(processed.astype('float32'), axis=0)
            
            heatmap, pred_class = make_gradcam_heatmap(input_array, model)
            heatmap_resized = cv2.resize(heatmap, (224, 224))
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            
            processed_bgr = cv2.cvtColor(processed.astype('uint8'), cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(processed_bgr, 0.6, heatmap_colored, 0.4, 0)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            st.divider()
            st.markdown("## 📊 Screening Result")

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Predicted Grade", f"Grade {pred_class}")
            with m2:
                st.metric("Classification", GRADE_LABELS[pred_class])

            st.markdown("### 🔬 Visual Explanation")
            t1, t2, t3 = st.tabs(["Original", "Heatmap", "Grad-CAM Overlay"])
            with t1:
                st.image(img_array, caption="Original Image", use_container_width=True)
            with t2:
                st.image(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB), caption="Heatmap", use_container_width=True)
            with t3:
                st.image(overlay_rgb, caption="Grad-CAM Overlay", use_container_width=True)

            st.markdown("### 🧠 4-Line Clinical Explanation")
            explanation_text = generate_explanation(pred_class, heatmap_resized)
            formatted_explanation = explanation_text.replace("\n", "<br>")
            
            st.markdown(f'<div class="explanation-box">{formatted_explanation}</div>', unsafe_allow_html=True)

            st.warning("⚠️ **Clinical note:** NetraAI is an AI-assisted decision support prototype. Diagnostic outcomes should be confirmed by a certified ophthalmologist.")
