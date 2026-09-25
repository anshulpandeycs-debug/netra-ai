import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="NETRA AI — DR Screening",
    page_icon="🩺",
    layout="wide"
)

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
# IMAGE PREPROCESSING & GRAD-CAM
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
    
    max_val = tf.math.reduce_max(heatmap)
    if float(max_val.numpy()) > 0:
        heatmap = heatmap / max_val

    confidence = float(tf.nn.softmax(preds[0])[pred_index].numpy())
    return heatmap.numpy(), int(pred_index.numpy()), confidence

# =========================================================
# DETAILED CLINICAL EXPLANATIONS & PREVENTION
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

    stage_details = {
        0: (
            "**Clinical Assessment:** No visible signs of diabetic retinopathy were identified across the retinal fundus image.\n\n"
            "**Pathological Findings:** The vascular structures, optic disc, and macula exhibit normal morphological characteristics without evidence of microaneurysms, intraretinal hemorrhages, or lipid exudation. The model's Grad-CAM salience map shows diffuse background activation, concentrating lightly in the "
            f"**{hot_region}** region without flagging focal microvascular lesions.\n\n"
            "**Prevention & Management Plan:**\n"
            "• Maintain strict blood glucose control with a target HbA1c below 7.0% to prevent microvascular stress.\n"
            "• Keep systemic blood pressure (<130/80 mmHg) and lipid profiles within optimal clinical ranges.\n"
            "• Continue routine annual dilated eye examinations to monitor long-term retinal health."
        ),
        1: (
            "**Clinical Assessment:** Mild Non-Proliferative Diabetic Retinopathy (NPDR) detected.\n\n"
            "**Pathological Findings:** Early microvascular changes are present, characterized predominantly by isolated microaneurysms—small saccular outpouchings in the capillary walls. High neural network salience is concentrated within the "
            f"**{hot_region}** quadrant, highlighting localized areas of early vessel wall weakening before significant structural fluid leakage occurs.\n\n"
            "**Prevention & Management Plan:**\n"
            "• Intensify glycemic monitoring to halt the progression of capillary basement membrane thickening.\n"
            "• Adopt a heart-healthy, low-sodium diet and engage in moderate aerobic exercise to support microvascular integrity.\n"
            "• Schedule a follow-up comprehensive ophthalmic evaluation within 9 to 12 months."
        ),
        2: (
            "**Clinical Assessment:** Moderate Non-Proliferative Diabetic Retinopathy (NPDR) detected.\n\n"
            "**Pathological Findings:** The fundus displays noticeable microvascular damage, including multiple microaneurysms, dot-and-blot intraretinal hemorrhages, and early hard exudates resulting from microvascular fluid leakage. The Grad-CAM visual attention is heavily concentrated in the "
            f"**{hot_region}** region, reflecting dense feature salience around localized areas of retinal edema and vascular breakdown.\n\n"
            "**Prevention & Management Plan:**\n"
            "• Optimize systemic metabolic parameters, focusing on tight glycemic control and aggressive blood pressure regulation.\n"
            "• Monitor closely for symptoms of macular involvement, such as central visual blurring or distortion.\n"
            "• Obtain a formal referral for an outpatient ophthalmology consultation within 3 to 6 months."
        ),
        3: (
            "**Clinical Assessment:** Severe Non-Proliferative Diabetic Retinopathy (NPDR) detected.\n\n"
            "**Pathological Findings:** Extensive microvascular occlusion is evident, marked by widespread intraretinal hemorrhages in all four quadrants, venous beading, or prominent intraretinal microvascular abnormalities (IRMA). Deep model activation clusters intensely in the "
            f"**{hot_region}** region, signaling significant retinal ischemia and substantial risk for progression to proliferative disease.\n\n"
            "**Prevention & Management Plan:**\n"
            "• Avoid strenuous physical activities or heavy weightlifting that could precipitate preretinal hemorrhaging.\n"
            "• Work closely with endocrinology and primary care teams to aggressively manage blood glucose, renal function, and blood pressure.\n"
            "• Seek urgent evaluation by a retina specialist within weeks for consideration of preventive interventions."
        ),
        4: (
            "**Clinical Assessment:** Proliferative Diabetic Retinopathy (PDR) detected.\n\n"
            "**Pathological Findings:** Advanced microvascular compromise is present, characterized by pathologic neovascularization (growth of abnormal, fragile new blood vessels) on the retina or optic disc. The neural network exhibits peak salience across high-risk vessel proliferation in the "
            f"**{hot_region}** region, highlighting lesions associated with severe risks of vitreous hemorrhage and tractional detachment.\n\n"
            "**Prevention & Management Plan:**\n"
            "• Seek immediate, specialized ophthalmic evaluation to explore anti-VEGF intravitreal injections or panretinal photocoagulation (PRP).\n"
            "• Avoid rapid posture shifts, head-down positions, or severe physical strain to minimize vessel rupture risks.\n"
            "• Maintain daily, rigorous glucose and blood pressure tracking under direct clinical supervision."
        )
    }

    return stage_details[grade]

GRADE_LABELS = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

# =========================================================
# MAIN APP INTERFACE
# =========================================================
st.title("🩺 NetraAI — Explainable AI for Diabetic Retinopathy Screening")
st.caption("Upload a retinal fundus image to get an AI-assisted DR grading with visual explanation.")

if not model_status:
    st.error("The model could not be loaded.")
    st.code(model_error)
    st.stop()

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

    st.markdown("### Clinical Explanation & Prevention Guidance")
    st.markdown(generate_explanation(pred_class, heatmap_resized))

    st.info("⚠️ AI-assisted screening tool — clinical evaluation should be performed by a qualified healthcare professional.")
