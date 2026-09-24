import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="NETRA AI — DR Screening", layout="wide")

@st.cache_resource
def load_netra_model():
    return tf.keras.models.load_model("netraai_final.keras")

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
    grad_model = tf.keras.models.Model(
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
        'superior-nasal': heatmap[:h//2, :w//2].mean(), 'superior-temporal': heatmap[:h//2, w//2:].mean(),
        'inferior-nasal': heatmap[h//2:, :w//2].mean(), 'inferior-temporal': heatmap[h//2:, w//2:].mean(),
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

st.title("🩺 NetraAI — Explainable AI for Diabetic Retinopathy Screening")
st.caption("Upload a retinal fundus image to get an AI-assisted DR grading with visual explanation.")

uploaded_file = st.file_uploader("Upload a retinal image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)

    with st.spinner("Analyzing..."):
        processed = preprocess_image(img_array, size=224)
        input_array = np.expand_dims(processed.astype('float32'), axis=0)
        heatmap, pred_class, confidence = make_gradcam_heatmap(input_array, model)
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(processed.astype('uint8'), cv2.COLOR_RGB2BGR), 0.6, heatmap_colored, 0.4, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img_array, caption="Original", use_container_width=True)
    with col2:
        st.image(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB), caption="Heatmap", use_container_width=True)
    with col3:
        st.image(overlay_rgb, caption="Grad-CAM Overlay", use_container_width=True)

    st.subheader(f"Predicted: {GRADE_LABELS[pred_class]}  (Grade {pred_class}/4)")
    st.write(f"**Confidence:** {confidence*100:.1f}%")

    st.markdown("### Clinical Explanation")
    st.write(generate_explanation(pred_class, heatmap_resized))

    st.info("⚠️ AI-assisted screening tool — clinical evaluation should be performed by a qualified healthcare professional.")
