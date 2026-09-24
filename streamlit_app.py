import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NetraAI — DR Screening",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LIGHT THEME + UI CSS
# ============================================================

st.markdown(
    """
<style>

/* =========================================================
   GLOBAL
   ========================================================= */

.stApp {
    background-color: #f7f9fc !important;
}

.main .block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Remove Streamlit decoration */
[data-testid="stDecoration"] {
    display: none;
}

/* Header */
header[data-testid="stHeader"] {
    background-color: #ffffff !important;
}

/* Toolbar */
[data-testid="stToolbar"] {
    background-color: #ffffff !important;
}

/* General text */
.stApp p,
.stApp label {
    color: #1e293b;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0;
}

section[data-testid="stSidebar"] * {
    color: #1e293b !important;
}

.sidebar-brand {
    font-size: 25px;
    font-weight: 750;
    color: #123b68 !important;
    margin-bottom: 4px;
}

.sidebar-subtitle {
    font-size: 13px;
    color: #64748b !important;
    line-height: 1.5;
}

.sidebar-heading {
    font-size: 14px;
    font-weight: 700;
    color: #123b68 !important;
    margin-top: 20px;
    margin-bottom: 8px;
}

.sidebar-item {
    font-size: 13px;
    color: #475569 !important;
    margin-bottom: 8px;
}


/* =========================================================
   MAIN HEADER
   ========================================================= */

.netra-header {
    background: linear-gradient(
        135deg,
        #ffffff 0%,
        #eef6ff 100%
    );

    border: 1px solid #dbe7f5;

    border-radius: 16px;

    padding: 26px 30px;

    margin-bottom: 28px;

    box-shadow:
        0 3px 12px rgba(15, 23, 42, 0.04);
}

.netra-title {
    font-size: 32px;
    font-weight: 750;
    color: #123b68 !important;
    margin: 0;
}

.netra-subtitle {
    font-size: 15px;
    color: #64748b !important;
    margin-top: 6px;
}


/* =========================================================
   SECTION TITLE
   ========================================================= */

.section-title {
    font-size: 21px;
    font-weight: 700;
    color: #123b68 !important;

    margin-top: 25px;
    margin-bottom: 12px;
}


/* =========================================================
   GENERAL CARD
   ========================================================= */

.netra-card {
    background-color: #ffffff;

    border: 1px solid #e2e8f0;

    border-radius: 14px;

    padding: 20px;

    box-shadow:
        0 2px 8px rgba(15, 23, 42, 0.04);

    margin-bottom: 18px;
}

.card-heading {
    font-size: 17px;
    font-weight: 700;
    color: #1e3a5f !important;

    margin-bottom: 8px;
}


/* =========================================================
   UPLOAD AREA
   ========================================================= */

[data-testid="stFileUploader"] {
    background-color: #ffffff !important;

    border: 1px solid #dbe4ef;

    border-radius: 14px;

    padding: 8px;
}

[data-testid="stFileUploader"] * {
    color: #1e293b !important;
}


/* =========================================================
   BUTTON
   ========================================================= */

.stButton > button {
    width: 100%;

    background-color: #1769aa !important;

    color: #ffffff !important;

    border: none !important;

    border-radius: 9px;

    padding: 0.65rem 1rem;

    font-size: 15px;

    font-weight: 650;

    transition: 0.2s;
}

.stButton > button:hover {
    background-color: #12588f !important;

    color: #ffffff !important;
}


/* =========================================================
   METRIC CARDS
   ========================================================= */

.metric-box {
    background-color: #ffffff;

    border: 1px solid #e2e8f0;

    border-radius: 13px;

    padding: 18px;

    text-align: center;

    min-height: 105px;

    box-shadow:
        0 2px 7px rgba(15, 23, 42, 0.03);
}

.metric-label {
    font-size: 13px;

    color: #64748b !important;

    margin-bottom: 7px;
}

.metric-value {
    font-size: 23px;

    font-weight: 750;

    color: #123b68 !important;
}


/* =========================================================
   RESULT CARD
   ========================================================= */

.result-card {
    background-color: #ffffff;

    border: 1px solid #dbe7f5;

    border-radius: 15px;

    padding: 24px;

    margin-top: 10px;

    box-shadow:
        0 2px 8px rgba(15, 23, 42, 0.04);
}

.prediction-label {
    font-size: 13px;

    font-weight: 600;

    color: #64748b !important;

    text-transform: uppercase;

    letter-spacing: 0.5px;
}

.prediction-value {
    font-size: 30px;

    font-weight: 750;

    color: #123b68 !important;

    margin-top: 4px;
}

.grade-badge {
    display: inline-block;

    background-color: #eaf3ff;

    color: #1769aa !important;

    border-radius: 20px;

    padding: 6px 13px;

    font-size: 13px;

    font-weight: 650;

    margin-top: 8px;
}


/* =========================================================
   IMAGE CARDS
   ========================================================= */

.image-card-title {
    font-size: 14px;

    font-weight: 650;

    color: #334155 !important;

    margin-bottom: 7px;
}


/* =========================================================
   EMPTY STATE
   ========================================================= */

.empty-state {
    background-color: #ffffff;

    border: 1px solid #e2e8f0;

    border-radius: 16px;

    padding: 48px 30px;

    text-align: center;

    margin-top: 20px;

    box-shadow:
        0 2px 8px rgba(15, 23, 42, 0.03);
}

.empty-icon {
    font-size: 48px;

    margin-bottom: 10px;
}

.empty-title {
    font-size: 21px;

    font-weight: 700;

    color: #123b68 !important;

    margin-bottom: 7px;
}

.empty-text {
    font-size: 14px;

    color: #64748b !important;
}


/* =========================================================
   INFO STRIP
   ========================================================= */

.info-strip {
    background-color: #eef6ff;

    border: 1px solid #d7e8fa;

    border-radius: 10px;

    padding: 13px 16px;

    color: #24547d !important;

    font-size: 13px;

    margin-top: 15px;
}


/* =========================================================
   FOOTER
   ========================================================= */

.netra-footer {
    text-align: center;

    color: #94a3b8 !important;

    font-size: 12px;

    padding-top: 30px;

    padding-bottom: 10px;
}


/* =========================================================
   TABS
   ========================================================= */

button[data-baseweb="tab"] {
    color: #475569 !important;

    font-weight: 600;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #1769aa !important;
}


/* =========================================================
   EXPANDER
   ========================================================= */

[data-testid="stExpander"] {
    background-color: #ffffff !important;

    border: 1px solid #e2e8f0 !important;

    border-radius: 12px !important;
}

[data-testid="stExpander"] * {
    color: #1e293b;
}


/* =========================================================
   PROGRESS BAR
   ========================================================= */

[data-testid="stProgress"] > div {
    background-color: #e2e8f0;
}

[data-testid="stProgress"] > div > div {
    background-color: #1769aa;
}


/* =========================================================
   ALERT BOX TEXT
   ========================================================= */

[data-testid="stAlert"] {
    border-radius: 10px;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_netra_model():

    return tf.keras.models.load_model(
        "netraai_final.keras"
    )


model = load_netra_model()


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_image(img_array, size=224):

    img = cv2.resize(
        img_array,
        (size, size)
    )

    lab = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge(
        (l, a, b)
    )

    return cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2RGB
    )


# ============================================================
# GRAD-CAM
# ============================================================

def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name="top_conv",
    pred_index=None
):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [
            model.get_layer(
                last_conv_layer_name
            ).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        last_conv_layer_output, preds = (
            grad_model(img_array)
        )

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

    heatmap = tf.squeeze(
        heatmap
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        heatmap
    )

    # Keep your existing prediction logic
    confidence = float(
        tf.nn.softmax(
            preds[0]
        )[pred_index].numpy()
    )

    return (
        heatmap.numpy(),
        int(pred_index.numpy()),
        confidence
    )


# ============================================================
# EXPLANATION
# ============================================================

def generate_explanation(
    grade,
    heatmap
):

    h, w = heatmap.shape

    quadrants = {

        "Superior-Nasal":
            heatmap[
                :h // 2,
                :w // 2
            ].mean(),

        "Superior-Temporal":
            heatmap[
                :h // 2,
                w // 2:
            ].mean(),

        "Inferior-Nasal":
            heatmap[
                h // 2:,
                :w // 2
            ].mean(),

        "Inferior-Temporal":
            heatmap[
                h // 2:,
                w // 2:
            ].mean()
    }

    hot_region = max(
        quadrants,
        key=quadrants.get
    )

    findings = {

        0:
        "The model classified this image as No DR. No strong model evidence associated with diabetic retinopathy was identified.",

        1:
        "The model classified this image as Mild DR. Early retinal changes associated with mild diabetic retinopathy may be present.",

        2:
        "The model classified this image as Moderate DR. Retinal changes associated with moderate diabetic retinopathy may be present.",

        3:
        "The model classified this image as Severe DR. More extensive retinal abnormalities may be present.",

        4:
        "The model classified this image as Proliferative DR. Abnormal retinal vascular changes may be present."
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


# ============================================================
# LABELS
# ============================================================

GRADE_LABELS = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            👁️ NetraAI
        </div>

        <div class="sidebar-subtitle">
            Explainable AI for Diabetic Retinopathy Screening
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        """
        <div class="sidebar-heading">
            📊 MODEL
        </div>

        <div class="sidebar-item">
            Architecture: Lightweight CNN
        </div>

        <div class="sidebar-item">
            Input: 224 × 224
        </div>

        <div class="sidebar-item">
            Classes: 5 DR grades
        </div>

        <div class="sidebar-item">
            Explainability: Grad-CAM
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        """
        <div class="sidebar-heading">
            🧪 DATASET
        </div>

        <div class="sidebar-item">
            APTOS 2019 Blindness Detection
        </div>

        <div class="sidebar-item">
            5-class DR grading
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        """
        <div class="sidebar-heading">
            ⚠️ IMPORTANT
        </div>

        <div class="sidebar-item">
            NetraAI is an AI-assisted screening
            prototype and does not replace
            professional clinical evaluation.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
<div class="netra-header">

    <div class="netra-title">
        👁️ NetraAI
    </div>

    <div class="netra-subtitle">
        Lightweight & Explainable AI for Diabetic Retinopathy Screening
    </div>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# RETINA ANALYSIS
# ============================================================

st.markdown(
    """
    <div class="section-title">
        🔍 Retina Analysis
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    "Upload a retinal fundus image to receive an "
    "AI-assisted DR classification and visual explanation."
)


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload retinal fundus image",
    type=[
        "png",
        "jpg",
        "jpeg"
    ],
    help="Supported formats: PNG, JPG and JPEG"
)


# ============================================================
# IF IMAGE UPLOADED
# ============================================================

if uploaded_file:

    img = Image.open(
        uploaded_file
    ).convert("RGB")

    img_array = np.array(
        img
    )

    # --------------------------------------------------------
    # IMAGE PREVIEW
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-title">
            📷 Image Preview
        </div>
        """,
        unsafe_allow_html=True
    )

    preview_col1, preview_col2 = st.columns(
        [2.5, 1]
    )

    with preview_col1:

        st.image(
            img_array,
            caption="Uploaded retinal fundus image",
            use_container_width=True
        )

    with preview_col2:

        st.markdown(
            f"""
<div class="netra-card">

    <div class="card-heading">
        Image Information
    </div>

    <p>
        <b>Filename:</b><br>
        {uploaded_file.name}
    </p>

    <p>
        <b>Width:</b> {img.width}px
    </p>

    <p>
        <b>Height:</b> {img.height}px
    </p>

    <p>
        <b>Processing:</b><br>
        224 × 224 pixels
    </p>

</div>
""",
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    st.write("")

    analyze = st.button(
        "🔬 Analyze Retina"
    )


    # ========================================================
    # ANALYSIS
    # ========================================================

    if analyze:

        with st.spinner(
            "NetraAI is analyzing the retinal image..."
        ):

            processed = preprocess_image(
                img_array,
                size=224
            )

            input_array = np.expand_dims(
                processed.astype(
                    "float32"
                ),
                axis=0
            )

            (
                heatmap,
                pred_class,
                confidence
            ) = make_gradcam_heatmap(
                input_array,
                model
            )

            heatmap_resized = cv2.resize(
                heatmap,
                (224, 224)
            )

            heatmap_colored = cv2.applyColorMap(
                np.uint8(
                    255 * heatmap_resized
                ),
                cv2.COLORMAP_JET
            )

            overlay = cv2.addWeighted(
                cv2.cvtColor(
                    processed.astype(
                        "uint8"
                    ),
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


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown(
            """
            <div class="section-title">
                📋 AI Screening Result
            </div>
            """,
            unsafe_allow_html=True
        )


        metric1, metric2, metric3 = st.columns(3)


        with metric1:

            st.markdown(
                f"""
<div class="metric-box">

    <div class="metric-label">
        PREDICTED GRADE
    </div>

    <div class="metric-value">
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

    <div class="metric-label">
        MODEL CONFIDENCE
    </div>

    <div class="metric-value">
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

    <div class="metric-label">
        CLASSIFICATION
    </div>

    <div class="metric-value"
         style="font-size:19px;">

        {GRADE_LABELS[pred_class]}

    </div>

</div>
""",
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        st.write("")

        st.write(
            "**Model confidence**"
        )

        st.progress(
            min(
                max(
                    confidence,
                    0.0
                ),
                1.0
            )
        )


        # ====================================================
        # EXPLAINABILITY
        # ====================================================

        st.markdown(
            """
            <div class="section-title">
                🧠 Explainability
            </div>
            """,
            unsafe_allow_html=True
        )


        image_col1, image_col2, image_col3 = st.columns(3)


        with image_col1:

            st.markdown(
                '<div class="image-card-title">Original Image</div>',
                unsafe_allow_html=True
            )

            st.image(
                img_array,
                use_container_width=True
            )


        with image_col2:

            st.markdown(
                '<div class="image-card-title">AI Attention Heatmap</div>',
                unsafe_allow_html=True
            )

            st.image(
                cv2.cvtColor(
                    heatmap_colored,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )


        with image_col3:

            st.markdown(
                '<div class="image-card-title">Grad-CAM Overlay</div>',
                unsafe_allow_html=True
            )

            st.image(
                overlay_rgb,
                use_container_width=True
            )


        # ====================================================
        # EXPLANATION
        # ====================================================

        (
            findings,
            hot_region,
            guidance
        ) = generate_explanation(
            pred_class,
            heatmap_resized
        )


        tab1, tab2, tab3 = st.tabs(
            [
                "📄 Result",
                "🧠 AI Explanation",
                "🏥 Clinical Guidance"
            ]
        )


        # ----------------------------------------------------
        # RESULT TAB
        # ----------------------------------------------------

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

    <div class="grade-badge">
        Grade {pred_class} / 4
    </div>

    <p style="margin-top:16px;">
        Model confidence:
        <b>
            {confidence * 100:.1f}%
        </b>
    </p>

</div>
""",
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # EXPLANATION TAB
        # ----------------------------------------------------

        with tab2:

            st.markdown(
                "### Model Observation"
            )

            st.write(
                findings
            )

            st.info(
                f"Grad-CAM attention was highest "
                f"in the **{hot_region}** region."
            )

            st.caption(
                "The heatmap shows image regions that "
                "contributed strongly to the model prediction. "
                "It should not be interpreted as a definitive "
                "clinical lesion map."
            )


        # ----------------------------------------------------
        # CLINICAL GUIDANCE TAB
        # ----------------------------------------------------

        with tab3:

            st.markdown(
                "### Screening Guidance"
            )

            st.write(
                guidance
            )

            st.warning(
                "This is an AI-assisted screening prototype. "
                "A qualified healthcare professional should "
                "make the final clinical assessment."
            )


        # ====================================================
        # TECHNICAL DETAILS
        # ====================================================

        with st.expander(
            "⚙️ View Technical Details"
        ):

            tech1, tech2, tech3 = st.columns(3)


            with tech1:

                st.markdown(
                    "**Input Resolution**"
                )

                st.write(
                    "224 × 224 pixels"
                )


            with tech2:

                st.markdown(
                    "**Preprocessing**"
                )

                st.write(
                    "Resize + CLAHE"
                )


            with tech3:

                st.markdown(
                    "**Explainability**"
                )

                st.write(
                    "Grad-CAM"
                )


            st.divider()

            st.markdown(
                "**Prediction Pipeline**"
            )

            st.code(
                """
Fundus Image
      ↓
Resize 224 × 224
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


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.markdown(
        """
<div class="empty-state">

    <div class="empty-icon">
        👁️
    </div>

    <div class="empty-title">
        Ready to analyze a retinal image
    </div>

    <div class="empty-text">
        Upload a fundus image above to start
        AI-assisted diabetic retinopathy screening.
    </div>

</div>
""",
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="netra-footer">

    NetraAI • Lightweight Explainable AI for DR Screening

    <br>

    Research / Prototype System

</div>
""",
    unsafe_allow_html=True
)
