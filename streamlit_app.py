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
# LIGHT UI STYLING
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f7f9fc;
    }

    /* Main content */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    /* Header */
    .netra-header {
        background: linear-gradient(135deg, #ffffff, #eef6ff);
        padding: 25px 30px;
        border-radius: 16px;
        border: 1px solid #dce7f5;
        margin-bottom: 25px;
    }

    .netra-title {
        font-size: 32px;
        font-weight: 700;
        color: #123b68;
        margin-bottom: 5px;
    }

    .netra-subtitle {
        font-size: 15px;
        color: #64748b;
    }

    /* Cards */
    .card {
        background: white;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        margin-bottom: 18px;
    }

    .card-title {
        font-size: 17px;
        font-weight: 650;
        color: #1e3a5f;
        margin-bottom: 10px;
    }

    /* Result card */
    .result-card {
        background: white;
        border-radius: 16px;
        border: 1px solid #dce7f5;
        padding: 24px;
        margin-top: 20px;
    }

    .prediction-label {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 5px;
    }

    .prediction-value {
        font-size: 30px;
        font-weight: 750;
        color: #123b68;
    }

    .grade-badge {
        display: inline-block;
        padding: 6px 13px;
        border-radius: 20px;
        background: #eaf3ff;
        color: #1769aa;
        font-weight: 600;
        font-size: 13px;
    }

    /* Metrics */
    .metric-box {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 17px;
        text-align: center;
    }

    .metric-number {
        font-size: 23px;
        font-weight: 700;
        color: #123b68;
    }

    .metric-label {
        color: #64748b;
        font-size: 13px;
    }

    /* Section title */
    .section-title {
        color: #123b68;
        font-size: 21px;
        font-weight: 700;
        margin-top: 28px;
        margin-bottom: 14px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
        padding-top: 25px;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background-color: #ffffff;
        border-radius: 14px;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 9px;
        border: none;
        background-color: #1769aa;
        color: white;
        font-weight: 600;
        padding: 0.65rem 1rem;
    }

    .stButton > button:hover {
        background-color: #12588f;
        color: white;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_netra_model():
    return tf.keras.models.load_model("netraai_final.keras")


model = load_netra_model()


# =========================================================
# PREPROCESSING
# =========================================================

def preprocess_image(img_array, size=224):

    img = cv2.resize(img_array, (size, size))

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))

    return cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2RGB
    )


# =========================================================
# GRAD-CAM
# =========================================================

def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name="top_conv",
    pred_index=None
):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        last_conv_layer_output, preds = grad_model(img_array)

        # Handle model output
        if pred_index is None:

            pred_index = tf.argmax(
                preds[0]
            )

        class_channel = preds[:, pred_index]

    grads = tape.gradient(
        class_channel,
        last_conv_layer_output
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    last_conv_layer_output = (
        last_conv_layer_output[0]
    )

    heatmap = (
        last_conv_layer_output
        @ pooled_grads[..., tf.newaxis]
    )

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(heatmap)

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        heatmap
    )

    confidence = float(
        tf.nn.softmax(preds[0])[pred_index].numpy()
    )

    return (
        heatmap.numpy(),
        int(pred_index.numpy()),
        confidence
    )


# =========================================================
# EXPLANATION
# =========================================================

def generate_explanation(grade, heatmap):

    h, w = heatmap.shape

    quadrants = {

        "Superior-Nasal":
            heatmap[:h//2, :w//2].mean(),

        "Superior-Temporal":
            heatmap[:h//2, w//2:].mean(),

        "Inferior-Nasal":
            heatmap[h//2:, :w//2].mean(),

        "Inferior-Temporal":
            heatmap[h//2:, w//2:].mean()
    }

    hot_region = max(
        quadrants,
        key=quadrants.get
    )

    findings = {

        0:
        "No visible diabetic retinopathy was detected. The retinal appearance is broadly consistent with the no-DR class.",

        1:
        "The model classified the image as Mild DR. Early retinal changes associated with mild diabetic retinopathy may be present.",

        2:
        "The model classified the image as Moderate DR. Retinal changes associated with moderate diabetic retinopathy may be present.",

        3:
        "The model classified the image as Severe DR. More extensive retinal abnormalities may be present.",

        4:
        "The model classified the image as Proliferative DR. Abnormal retinal vascular changes may be present."
    }

    guidance = {

        0:
        "Routine eye screening and appropriate diabetes management are recommended.",

        1:
        "Follow-up screening should be performed according to the patient's clinical risk and healthcare provider's recommendation.",

        2:
        "Professional ophthalmic evaluation is recommended.",

        3:
        "Prompt ophthalmic evaluation is recommended.",

        4:
        "Urgent evaluation by an ophthalmologist or retina specialist is recommended."
    }

    return (
        findings[grade],
        hot_region,
        guidance[grade]
    )


# =========================================================
# LABELS
# =========================================================

GRADE_LABELS = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]

GRADE_COLORS = [
    "#2e7d32",
    "#7b7b00",
    "#e67e22",
    "#d35400",
    "#b71c1c"
]


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "## 👁️ NetraAI"
    )

    st.caption(
        "Explainable AI for Diabetic Retinopathy Screening"
    )

    st.divider()

    st.markdown("### 📊 Model")

    st.write(
        "**Architecture:** Lightweight CNN"
    )

    st.write(
        "**Input:** 224 × 224 retinal image"
    )

    st.write(
        "**Classes:** 5 DR grades"
    )

    st.write(
        "**Explainability:** Grad-CAM"
    )

    st.divider()

    st.markdown("### 🧪 Dataset")

    st.write(
        "**APTOS 2019 Blindness Detection**"
    )

    st.write(
        "5-class diabetic retinopathy grading"
    )

    st.divider()

    st.markdown("### ⚠️ Important")

    st.caption(
        "NetraAI is an AI-assisted screening "
        "prototype and should not replace "
        "professional clinical evaluation."
    )


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="netra-header">

<div class="netra-title">
👁️ NetraAI
</div>

<div class="netra-subtitle">
Lightweight & Explainable AI for Diabetic Retinopathy Screening
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# INTRO
# =========================================================

st.markdown(
    "### 🔍 Retina Analysis"
)

st.write(
    "Upload a retinal fundus image and NetraAI "
    "will classify the image into one of five DR grades "
    "and provide a Grad-CAM visual explanation."
)


# =========================================================
# UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload retinal fundus image",
    type=["png", "jpg", "jpeg"],
    help="Supported formats: PNG, JPG and JPEG"
)


if uploaded_file:

    img = Image.open(
        uploaded_file
    ).convert("RGB")

    img_array = np.array(img)

    # -----------------------------------------------------
    # IMAGE PREVIEW
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📷 Image Preview</div>',
        unsafe_allow_html=True
    )

    preview_col1, preview_col2 = st.columns(
        [2, 1]
    )

    with preview_col1:

        st.image(
            img_array,
            caption="Uploaded retinal fundus image",
            use_container_width=True
        )

    with preview_col2:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="card-title">Image Information</div>',
            unsafe_allow_html=True
        )

        st.write(
            f"**Filename:** {uploaded_file.name}"
        )

        st.write(
            f"**Width:** {img.width}px"
        )

        st.write(
            f"**Height:** {img.height}px"
        )

        st.write(
            "**Processing:** 224 × 224"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    # -----------------------------------------------------
    # ANALYZE BUTTON
    # -----------------------------------------------------

    analyze = st.button(
        "🔬 Analyze Retina"
    )

    if analyze:

        with st.spinner(
            "NetraAI is analyzing the retinal image..."
        ):

            processed = preprocess_image(
                img_array,
                size=224
            )

            input_array = np.expand_dims(
                processed.astype("float32"),
                axis=0
            )

            heatmap, pred_class, confidence = (
                make_gradcam_heatmap(
                    input_array,
                    model
                )
            )

            heatmap_resized = cv2.resize(
                heatmap,
                (224, 224)
            )

            heatmap_colored = cv2.applyColorMap(
                np.uint8(255 * heatmap_resized),
                cv2.COLORMAP_JET
            )

            overlay = cv2.addWeighted(
                cv2.cvtColor(
                    processed.astype("uint8"),
                    cv2.COLOR_RGB2BGR
                ),
                0.6,
                heatmap_colored,
                0.4,
                0
            )

            overlay_rgb = cv2.cvtColor(
                overlay,
                cv2.COLOR_BGR2RGB
            )

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.markdown(
            '<div class="section-title">📋 AI Screening Result</div>',
            unsafe_allow_html=True
        )

        metric1, metric2, metric3 = st.columns(3)

        with metric1:

            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Predicted Grade</div>
                    <div class="metric-number">
                        Grade {pred_class}/4
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with metric2:

            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-number">
                        {confidence * 100:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with metric3:

            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Classification</div>
                    <div class="metric-number"
                         style="font-size:19px;">
                        {GRADE_LABELS[pred_class]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Confidence bar

        st.write("")

        st.write(
            "**Model confidence**"
        )

        st.progress(
            min(confidence, 1.0)
        )

        # -------------------------------------------------
        # IMAGE EXPLANATION
        # -------------------------------------------------

        st.markdown(
            '<div class="section-title">🧠 Explainability</div>',
            unsafe_allow_html=True
        )

        image_col1, image_col2, image_col3 = st.columns(3)

        with image_col1:

            st.image(
                img_array,
                caption="Original Image",
                use_container_width=True
            )

        with image_col2:

            st.image(
                cv2.cvtColor(
                    heatmap_colored,
                    cv2.COLOR_BGR2RGB
                ),
                caption="AI Attention Heatmap",
                use_container_width=True
            )

        with image_col3:

            st.image(
                overlay_rgb,
                caption="Grad-CAM Overlay",
                use_container_width=True
            )

        # -------------------------------------------------
        # EXPLANATION
        # -------------------------------------------------

        findings, hot_region, guidance = (
            generate_explanation(
                pred_class,
                heatmap_resized
            )
        )

        tab1, tab2, tab3 = st.tabs(
            [
                "📄 Result",
                "🧠 AI Explanation",
                "🏥 Clinical Guidance"
            ]
        )

        with tab1:

            st.markdown(
                f"""
                <div class="result-card">

                    <div class="prediction-label">
                        AI PREDICTION
                    </div>

                    <div class="prediction-value">
                        {GRADE_LABELS[pred_class]}
                    </div>

                    <br>

                    <span class="grade-badge">
                        Grade {pred_class} / 4
                    </span>

                    <p style="margin-top:15px;">
                        Model confidence:
                        <b>{confidence * 100:.1f}%</b>
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        with tab2:

            st.markdown(
                f"**Model observation**"
            )

            st.write(
                findings
            )

            st.info(
                f"Grad-CAM attention was highest "
                f"in the **{hot_region}** region."
            )

            st.caption(
                "The heatmap represents regions that "
                "contributed strongly to the model's prediction. "
                "It should not be interpreted as a definitive "
                "clinical lesion map."
            )

        with tab3:

            st.write(
                guidance
            )

            st.warning(
                "This output is for AI-assisted screening "
                "and educational/research purposes. "
                "A qualified healthcare professional should "
                "make the clinical diagnosis."
            )

        # -------------------------------------------------
        # TECHNICAL DETAILS
        # -------------------------------------------------

        with st.expander(
            "⚙️ View Technical Details"
        ):

            tech1, tech2, tech3 = st.columns(3)

            with tech1:

                st.write(
                    "**Input Resolution**"
                )

                st.write(
                    "224 × 224 pixels"
                )

            with tech2:

                st.write(
                    "**Preprocessing**"
                )

                st.write(
                    "Resize + CLAHE"
                )

            with tech3:

                st.write(
                    "**Explainability**"
                )

                st.write(
                    "Grad-CAM"
                )

            st.write("")

            st.write(
                "**Prediction pipeline:**"
            )

            st.code(
                """
Fundus Image
      ↓
Resize 224×224
      ↓
CLAHE Enhancement
      ↓
Lightweight CNN
      ↓
DR Classification
      ↓
Grad-CAM
      ↓
Prediction + Explanation
                """,
                language="text"
            )


# =========================================================
# EMPTY STATE
# =========================================================

else:

    st.markdown(
        """
        <div class="card"
             style="text-align:center;
                    padding:45px;">

            <div style="font-size:45px;">
                👁️
            </div>

            <h3 style="color:#123b68;">
                Ready to analyze a retinal image
            </h3>

            <p style="color:#64748b;">
                Upload a fundus image above to start
                AI-assisted diabetic retinopathy screening.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        NetraAI • Lightweight Explainable AI for DR Screening
        <br>
        Research / Prototype System
    </div>
    """,
    unsafe_allow_html=True
)
