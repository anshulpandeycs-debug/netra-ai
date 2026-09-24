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
# LIGHT THEME / SIMPLE CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #f7fafb;
    }

    /* Main content width */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    /* Headings */
    h1, h2, h3 {
        color: #123b4a !important;
    }

    /* Normal text */
    p, label {
        color: #526572;
    }

    /* Upload box */
    [data-testid="stFileUploader"] {
        background-color: #ffffff;
        border: 1px dashed #9bbfc3;
        border-radius: 12px;
        padding: 10px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #0f766e;
        background-color: #0f766e;
        color: white;
        font-weight: 600;
        padding: 0.55rem 1.2rem;
    }

    .stButton > button:hover {
        background-color: #115e59;
        border-color: #115e59;
        color: white;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e1e8ea;
        border-radius: 12px;
        padding: 15px;
    }

    /* Info boxes */
    .info-box {
        background-color: #ffffff;
        border: 1px solid #dce7e9;
        border-radius: 12px;
        padding: 18px;
        margin: 10px 0;
    }

    .result-box {
        background-color: #ffffff;
        border: 1px solid #dce7e9;
        border-radius: 14px;
        padding: 22px;
        margin-top: 15px;
    }

    .small-text {
        font-size: 13px;
        color: #64748b;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# MODEL
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
# PREPROCESSING
# =========================================================

def preprocess_image(img_array, size=224):

    img = cv2.resize(img_array, (size, size))

    # Convert RGB -> LAB
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    l, a, b = cv2.split(lab)

    # Improve retinal image contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))

    processed = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2RGB
    )

    return processed


# =========================================================
# GRAD-CAM
# =========================================================

def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name="top_conv",
    pred_index=None
):

    try:

        grad_model = tf.keras.models.Model(
            [model.inputs],
            [
                model.get_layer(
                    last_conv_layer_name
                ).output,
                model.output
            ]
        )

    except Exception:

        # Automatically find a convolution layer
        conv_layers = [
            layer for layer in model.layers
            if isinstance(
                layer,
                tf.keras.layers.Conv2D
            )
        ]

        if not conv_layers:
            raise ValueError(
                "No convolution layer found for Grad-CAM."
            )

        last_conv_layer = conv_layers[-1]

        grad_model = tf.keras.models.Model(
            [model.inputs],
            [
                last_conv_layer.output,
                model.output
            ]
        )

    with tf.GradientTape() as tape:

        last_conv_layer_output, preds = grad_model(
            img_array
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

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(heatmap)

    if float(max_value.numpy()) > 0:
        heatmap = heatmap / max_value

    # Confidence
    probabilities = tf.nn.softmax(
        preds[0]
    )

    confidence = float(
        probabilities[pred_index].numpy()
    )

    return (
        heatmap.numpy(),
        int(pred_index.numpy()),
        confidence
    )


# =========================================================
# EXPLANATION
# =========================================================

def generate_explanation(
    grade,
    heatmap
):

    h, w = heatmap.shape

    quadrants = {

        "superior-nasal":
            heatmap[
                :h // 2,
                :w // 2
            ].mean(),

        "superior-temporal":
            heatmap[
                :h // 2,
                w // 2:
            ].mean(),

        "inferior-nasal":
            heatmap[
                h // 2:,
                :w // 2
            ].mean(),

        "inferior-temporal":
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
        "No visible diabetic retinopathy was detected. "
        "The retinal appearance is consistent with the No DR category.",

        1:
        "The model classified the image as Mild NPDR. "
        "Early retinal changes such as microaneurysm-like features "
        "may be present.",

        2:
        "The model classified the image as Moderate NPDR. "
        "Features associated with microvascular retinal changes "
        "may be present.",

        3:
        "The model classified the image as Severe NPDR. "
        "More extensive retinal abnormalities may be present.",

        4:
        "The model classified the image as Proliferative DR. "
        "Features associated with advanced retinal changes "
        "may be present."
    }

    guidance = {

        0:
        "Routine diabetic eye screening and appropriate "
        "blood-glucose and blood-pressure management are advised.",

        1:
        "Clinical follow-up and continued diabetic eye screening "
        "are recommended.",

        2:
        "Professional ophthalmic evaluation is recommended "
        "to determine appropriate follow-up.",

        3:
        "Prompt ophthalmic evaluation is recommended.",

        4:
        "Urgent evaluation by an eye-care professional is recommended."
    }

    return (
        findings[grade]
        + f"\n\nThe Grad-CAM attention was concentrated mainly "
        f"in the **{hot_region}** region."
        + f"\n\n{guidance[grade]}"
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


GRADE_DESCRIPTIONS = [
    "No visible signs of diabetic retinopathy",
    "Early retinal changes",
    "Moderate retinal changes",
    "Advanced non-proliferative changes",
    "Advanced proliferative changes"
]


# =========================================================
# SESSION STATE
# =========================================================

if "result" not in st.session_state:
    st.session_state.result = None

if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 👁️ NetraAI")

    st.caption(
        "Explainable AI for Diabetic Retinopathy Screening"
    )

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

    st.caption(
        "NetraAI provides AI-assisted screening "
        "and should not replace professional clinical evaluation."
    )


# =========================================================
# HEADER
# =========================================================

st.title("NetraAI")

st.subheader(
    "Diabetic Retinopathy Screening"
)

st.write(
    "Upload a retinal fundus image to obtain an "
    "AI-assisted DR classification with a visual explanation."
)


# =========================================================
# MODEL ERROR
# =========================================================

if not model_status:

    st.error(
        "The model could not be loaded."
    )

    st.code(
        model_error
    )

    st.stop()


# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown("### 📤 Upload Retinal Image")

uploaded_file = st.file_uploader(
    "Choose a retinal fundus image",
    type=["png", "jpg", "jpeg"],
    help="Supported formats: PNG, JPG and JPEG"
)


# =========================================================
# BEFORE UPLOAD
# =========================================================

if uploaded_file is None:

    st.info(
        "👁️ **Ready for screening**\n\n"
        "Upload a clear retinal fundus photograph above "
        "to begin the analysis."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Input",
            "224 × 224",
            "retinal image"
        )

    with col2:
        st.metric(
            "DR Classes",
            "5",
            "Grade 0–4"
        )

    with col3:
        st.metric(
            "Explanation",
            "Grad-CAM",
            "visual attention"
        )

    st.stop()


# =========================================================
# LOAD IMAGE
# =========================================================

img = Image.open(
    uploaded_file
).convert("RGB")

img_array = np.array(img)


# =========================================================
# IMAGE PREVIEW
# =========================================================

st.markdown("### 🖼️ Image Preview")

preview_col, details_col = st.columns(
    [1.5, 1]
)

with preview_col:

    st.image(
        img_array,
        caption="Uploaded retinal fundus image",
        use_container_width=True
    )


with details_col:

    st.markdown("#### Image Information")

    st.write(
        f"**File:** {uploaded_file.name}"
    )

    st.write(
        f"**Resolution:** {img_array.shape[1]} × "
        f"{img_array.shape[0]} px"
    )

    st.write(
        f"**Format:** {uploaded_file.type}"
    )

    st.info(
        "For best results, use a clear fundus photograph "
        "with the retina centered in the image."
    )


# =========================================================
# ACTION BUTTONS
# =========================================================

button_col1, button_col2 = st.columns(
    [1, 4]
)

with button_col1:

    analyze_button = st.button(
        "🔍 Analyze Image",
        use_container_width=True
    )

with button_col2:

    clear_button = st.button(
        "Clear Result",
        use_container_width=False
    )


if clear_button:

    st.session_state.result = None
    st.rerun()


# =========================================================
# ANALYSIS
# =========================================================

if analyze_button:

    with st.spinner(
        "Analyzing retinal image..."
    ):

        try:

            # Preprocess
            processed = preprocess_image(
                img_array,
                size=224
            )

            input_array = np.expand_dims(
                processed.astype("float32"),
                axis=0
            )

            # Grad-CAM + prediction
            heatmap, pred_class, confidence = (
                make_gradcam_heatmap(
                    input_array,
                    model
                )
            )

            # Resize heatmap
            heatmap_resized = cv2.resize(
                heatmap,
                (224, 224)
            )

            # Generate colored heatmap
            heatmap_colored = cv2.applyColorMap(
                np.uint8(
                    255 * heatmap_resized
                ),
                cv2.COLORMAP_JET
            )

            # Create overlay
            processed_bgr = cv2.cvtColor(
                processed.astype("uint8"),
                cv2.COLOR_RGB2BGR
            )

            overlay = cv2.addWeighted(
                processed_bgr,
                0.6,
                heatmap_colored,
                0.4,
                0
            )

            overlay_rgb = cv2.cvtColor(
                overlay,
                cv2.COLOR_BGR2RGB
            )

            explanation = generate_explanation(
                pred_class,
                heatmap_resized
            )

            st.session_state.result = {
                "grade": pred_class,
                "confidence": confidence,
                "heatmap": heatmap_colored,
                "overlay": overlay_rgb,
                "explanation": explanation
            }

        except Exception as e:

            st.error(
                "An error occurred while analyzing the image."
            )

            st.exception(e)


# =========================================================
# RESULTS
# =========================================================

if st.session_state.result is not None:

    result = st.session_state.result

    grade = result["grade"]
    confidence = result["confidence"]

    st.divider()

    st.markdown("## 📊 Screening Result")

    # Metrics
    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "Predicted Grade",
            f"Grade {grade}"
        )

    with metric2:

        st.metric(
            "Classification",
            GRADE_LABELS[grade]
        )

    with metric3:

        st.metric(
            "Model Confidence",
            f"{confidence * 100:.1f}%"
        )


    # Result description
    st.markdown(
        f"""
        <div class="result-box">
        <h3>{GRADE_LABELS[grade]}</h3>
        <p>{GRADE_DESCRIPTIONS[grade]}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # VISUAL EXPLANATION
    # =====================================================

    st.markdown("### 🔬 Visual Explanation")

    tab1, tab2, tab3 = st.tabs(
        [
            "Original",
            "Heatmap",
            "Grad-CAM Overlay"
        ]
    )

    with tab1:

        st.image(
            img_array,
            caption="Original retinal image",
            use_container_width=True
        )

    with tab2:

        st.image(
            cv2.cvtColor(
                result["heatmap"],
                cv2.COLOR_BGR2RGB
            ),
            caption="Model attention heatmap",
            use_container_width=True
        )

    with tab3:

        st.image(
            result["overlay"],
            caption="Grad-CAM explanation",
            use_container_width=True
        )


    # =====================================================
    # EXPLANATION
    # =====================================================

    st.markdown("### 🧠 Why did the model make this prediction?")

    st.info(
        result["explanation"]
    )


    # =====================================================
    # SIMPLE GRADE SCALE
    # =====================================================

    st.markdown("### 📈 DR Grade Scale")

    grade_cols = st.columns(5)

    for i, col in enumerate(grade_cols):

        with col:

            if i == grade:

                st.success(
                    f"**Grade {i}**\n\n"
                    f"{GRADE_LABELS[i]}"
                )

            else:

                st.markdown(
                    f"**Grade {i}**\n\n"
                    f"{GRADE_LABELS[i]}"
                )


    # =====================================================
    # DISCLAIMER
    # =====================================================

    st.warning(
        "⚠️ **Clinical note:** NetraAI is an AI-assisted "
        "screening prototype. The result should not be treated "
        "as a definitive medical diagnosis. Clinical evaluation "
        "by a qualified healthcare professional is recommended."
    )
