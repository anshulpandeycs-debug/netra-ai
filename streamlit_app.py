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
    page_icon="NETRA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# SESSION STATE
# =========================================================

if "account_created" not in st.session_state:
    st.session_state.account_created = False

if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""

if "patient_id" not in st.session_state:
    st.session_state.patient_id = ""

if "mobile" not in st.session_state:
    st.session_state.mobile = ""

if "dob" not in st.session_state:
    st.session_state.dob = None

if "active_page" not in st.session_state:
    st.session_state.active_page = "New Screening"

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* ================================
       GLOBAL
       ================================ */

    .stApp {
        background: #f8fafc;
    }

    /* ================================
       SIDEBAR
       ================================ */

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    /* NETRA BRAND */

    .netra-brand {
        font-size: 30px;
        font-weight: 900;
        letter-spacing: 1px;
        color: #173f8a;
        margin-bottom: 3px;
        line-height: 1.1;
    }

    .netra-subtitle {
        font-size: 12px;
        color: #64748b;
        margin-bottom: 25px;
        letter-spacing: 0.5px;
    }

    /* SIDEBAR BUTTONS */

    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        border: 1px solid transparent;
        border-radius: 10px;
        background: transparent;
        color: #334155;
        text-align: left;
        font-weight: 600;
        padding: 11px 14px;
        margin: 3px 0;
        transition: all 0.18s ease;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #eaf2ff;
        color: #174ea6;
        border-color: #c7dcff;
    }

    /* ACTIVE BUTTON */

    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: #174ea6;
        color: white;
        border-color: #174ea6;
    }

    /* ACCOUNT CARD */

    .account-card {
        background: #f1f6ff;
        border: 1px solid #d9e7ff;
        border-radius: 12px;
        padding: 14px;
        margin: 10px 0 20px 0;
    }

    .account-label {
        font-size: 11px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .account-name {
        font-size: 17px;
        font-weight: 750;
        color: #173f8a;
        margin-top: 3px;
    }

    .account-id {
        font-size: 12px;
        color: #64748b;
        margin-top: 4px;
    }

    /* ================================
       MAIN CONTENT
       ================================ */

    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #172554;
        margin-bottom: 5px;
    }

    .main-subtitle {
        color: #64748b;
        font-size: 15px;
        margin-bottom: 25px;
    }

    /* PATIENT BANNER */

    .patient-banner {
        background: linear-gradient(
            90deg,
            #eef5ff,
            #f8fbff
        );
        border: 1px solid #d8e7ff;
        border-radius: 14px;
        padding: 15px 18px;
        margin-bottom: 20px;
    }

    .patient-banner-title {
        color: #173f8a;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .patient-banner-name {
        font-size: 21px;
        font-weight: 800;
        color: #172554;
    }

    /* ALERT */

    .patient-alert {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #9a3412;
        padding: 14px 16px;
        border-radius: 10px;
        margin: 10px 0 18px 0;
        font-weight: 600;
    }

    /* RESULT */

    .result-card {
        padding: 18px;
        border-radius: 14px;
        background: white;
        border: 1px solid #e2e8f0;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
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
# IMAGE PREPROCESSING
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

        conv_layers = [
            layer
            for layer in model.layers
            if isinstance(
                layer,
                tf.keras.layers.Conv2D
            )
        ]

        grad_model = tf.keras.models.Model(
            [model.inputs],
            [
                conv_layers[-1].output,
                model.output
            ]
        )

    with tf.GradientTape() as tape:

        last_conv_layer_output, preds = grad_model(
            img_array
        )

        if pred_index is None:
            pred_index = tf.argmax(preds[0])

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

    max_val = tf.math.reduce_max(
        heatmap
    )

    if float(max_val.numpy()) > 0:

        heatmap = (
            heatmap / max_val
        )

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


# =========================================================
# CLINICAL EXPLANATION
# =========================================================

def generate_explanation(
    grade,
    heatmap
):

    h, w = heatmap.shape

    quadrants = {

        "superior-nasal":
            heatmap[:h//2, :w//2].mean(),

        "superior-temporal":
            heatmap[:h//2, w//2:].mean(),

        "inferior-nasal":
            heatmap[h//2:, :w//2].mean(),

        "inferior-temporal":
            heatmap[h//2:, w//2:].mean()
    }

    hot_region = max(
        quadrants,
        key=quadrants.get
    )

    stage_details = {

        0:
        (
            "**Clinical Assessment:** "
            "No visible signs of diabetic "
            "retinopathy were identified "
            "across the retinal fundus image.\n\n"

            "**Pathological Findings:** "
            "The vascular structures, optic disc, "
            "and macula exhibit normal "
            "morphological characteristics "
            "without evidence of microaneurysms, "
            "intraretinal hemorrhages, "
            "or lipid exudation.\n\n"

            "**Prevention & Management Plan:**\n"
            "• Maintain appropriate blood glucose control.\n"
            "• Monitor blood pressure and lipid levels.\n"
            "• Continue routine eye examinations."
        ),

        1:
        (
            "**Clinical Assessment:** "
            "Mild Non-Proliferative Diabetic "
            "Retinopathy (NPDR) detected.\n\n"

            "**Pathological Findings:** "
            "Early microvascular changes may be "
            "present, including isolated "
            "microaneurysm-like features. "
            f"Model attention is concentrated in the "
            f"**{hot_region}** region.\n\n"

            "**Prevention & Management Plan:**\n"
            "• Maintain appropriate glycemic control.\n"
            "• Follow a healthy lifestyle.\n"
            "• Schedule appropriate ophthalmic follow-up."
        ),

        2:
        (
            "**Clinical Assessment:** "
            "Moderate Non-Proliferative "
            "Diabetic Retinopathy (NPDR) detected.\n\n"

            "**Pathological Findings:** "
            "The image contains features that may "
            "correspond with increased retinal "
            "microvascular abnormalities. "
            f"Model attention is concentrated in the "
            f"**{hot_region}** region.\n\n"

            "**Management:**\n"
            "• Maintain appropriate metabolic control.\n"
            "• Monitor for progression.\n"
            "• Obtain professional ophthalmic review."
        ),

        3:
        (
            "**Clinical Assessment:** "
            "Severe Non-Proliferative Diabetic "
            "Retinopathy (NPDR) detected.\n\n"

            "**Pathological Findings:** "
            "The model identifies extensive "
            "retinal features associated with "
            "advanced disease. "
            f"Attention is concentrated in the "
            f"**{hot_region}** region.\n\n"

            "**Management:**\n"
            "• Prompt ophthalmic evaluation is recommended.\n"
            "• Maintain appropriate systemic disease control.\n"
            "• Follow specialist recommendations."
        ),

        4:
        (
            "**Clinical Assessment:** "
            "Proliferative Diabetic Retinopathy "
            "(PDR) detected.\n\n"

            "**Pathological Findings:** "
            "The model identifies features "
            "associated with advanced retinal "
            "vascular disease. "
            f"Attention is concentrated in the "
            f"**{hot_region}** region.\n\n"

            "**Management:**\n"
            "• Seek prompt specialist ophthalmic evaluation.\n"
            "• Follow clinician-directed treatment.\n"
            "• Maintain appropriate glucose and blood-pressure management."
        )
    }

    return stage_details[grade]


GRADE_LABELS = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="netra-brand">
            NETRA AI
        </div>

        <div class="netra-subtitle">
            Explainable Retinal Screening
        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------
    # ACCOUNT
    # -----------------------------------------

    if st.session_state.account_created:

        st.markdown(
            f"""
            <div class="account-card">

                <div class="account-label">
                    Current Patient
                </div>

                <div class="account-name">
                    {st.session_state.patient_name}
                </div>

                <div class="account-id">
                    NETRA ID:
                    {st.session_state.patient_id}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            "### Create Patient Account"
        )

        name = st.text_input(
            "Patient Name",
            placeholder="Enter patient's full name",
            key="registration_name"
        )

        mobile = st.text_input(
            "Mobile Number",
            placeholder="Enter mobile number",
            max_chars=10,
            key="registration_mobile"
        )

        dob = st.date_input(
            "Date of Birth",
            value=None,
            key="registration_dob"
        )

        if st.button(
            "Create Account",
            type="primary",
            use_container_width=True
        ):

            if not name.strip():

                st.error(
                    "Please enter the patient's name."
                )

            elif mobile and (
                not mobile.isdigit()
                or len(mobile) != 10
            ):

                st.error(
                    "Please enter a valid 10-digit mobile number."
                )

            else:

                import hashlib

                patient_seed = (
                    name.strip()
                    + str(mobile)
                )

                patient_id = (
                    "NETRA-"
                    + hashlib.sha256(
                        patient_seed.encode()
                    ).hexdigest()[:8].upper()
                )

                st.session_state.patient_name = (
                    name.strip()
                )

                st.session_state.patient_id = (
                    patient_id
                )

                st.session_state.mobile = (
                    mobile
                )

                st.session_state.dob = (
                    dob
                )

                st.session_state.account_created = True

                st.success(
                    "Patient account created."
                )

                st.rerun()

    st.markdown("---")

    # -----------------------------------------
    # NAVIGATION
    # -----------------------------------------

    st.markdown(
        "**NAVIGATION**"
    )

    if st.button(
        "Dashboard",
        use_container_width=True,
        type=(
            "primary"
            if st.session_state.active_page == "Dashboard"
            else "secondary"
        )
    ):

        st.session_state.active_page = "Dashboard"
        st.rerun()

    if st.button(
        "New Screening",
        use_container_width=True,
        type=(
            "primary"
            if st.session_state.active_page == "New Screening"
            else "secondary"
        )
    ):

        st.session_state.active_page = "New Screening"
        st.rerun()

    if st.button(
        "Screening History",
        use_container_width=True,
        type=(
            "primary"
            if st.session_state.active_page == "Screening History"
            else "secondary"
        )
    ):

        st.session_state.active_page = "Screening History"
        st.rerun()

    if st.button(
        "Model Insights",
        use_container_width=True,
        type=(
            "primary"
            if st.session_state.active_page == "Model Insights"
            else "secondary"
        )
    ):

        st.session_state.active_page = "Model Insights"
        st.rerun()

    if st.button(
        "About",
        use_container_width=True,
        type=(
            "primary"
            if st.session_state.active_page == "About"
            else "secondary"
        )
    ):

        st.session_state.active_page = "About"
        st.rerun()

# =========================================================
# MODEL ERROR
# =========================================================

if not model_status:

    st.error(
        "The NETRA AI model could not be loaded."
    )

    st.code(model_error)

    st.stop()

# =========================================================
# DASHBOARD
# =========================================================

if st.session_state.active_page == "Dashboard":

    st.markdown(
        '<div class="main-title">NETRA AI Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-subtitle">'
        'Explainable AI-assisted diabetic retinopathy screening'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "System Status",
            "ONLINE"
        )

    with c2:
        st.metric(
            "AI Model",
            "READY"
        )

    with c3:
        st.metric(
            "Screening Engine",
            "ACTIVE"
        )

    st.markdown("---")

    if st.session_state.account_created:

        st.success(
            f"Patient ready for screening: "
            f"{st.session_state.patient_name}"
        )

    else:

        st.info(
            "Create a patient account from the sidebar "
            "before starting a screening."
        )

# =========================================================
# NEW SCREENING
# =========================================================

elif st.session_state.active_page == "New Screening":

    st.markdown(
        '<div class="main-title">New Screening</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-subtitle">'
        'Upload a retinal fundus image for AI-assisted analysis.'
        '</div>',
        unsafe_allow_html=True
    )

    # -----------------------------------------
    # PATIENT REQUIREMENT
    # -----------------------------------------

    if not st.session_state.account_created:

        st.markdown(
            """
            <div class="patient-alert">
                Patient registration required — please create a
                patient account from the sidebar before uploading
                a retinal image.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.warning(
            "Please fill the patient's name and create the "
            "patient account before starting screening."
        )

        st.stop()

    # -----------------------------------------
    # PATIENT BANNER
    # -----------------------------------------

    st.markdown(
        f"""
        <div class="patient-banner">

            <div class="patient-banner-title">
                Screening Patient
            </div>

            <div class="patient-banner-name">
                {st.session_state.patient_name}
            </div>

            <div>
                NETRA ID:
                <b>{st.session_state.patient_id}</b>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload a retinal image",
        type=[
            "png",
            "jpg",
            "jpeg"
        ]
    )

    if uploaded_file:

        img = Image.open(
            uploaded_file
        ).convert("RGB")

        img_array = np.array(img)

        with st.spinner(
            "NETRA AI is analyzing the retinal image..."
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
            np.uint8(
                255 * heatmap_resized
            ),
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

        st.markdown(
            "### Explainable AI Analysis"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.image(
                img_array,
                caption="Original Retinal Image",
                use_container_width=True
            )

        with col2:

            st.image(
                cv2.cvtColor(
                    heatmap_colored,
                    cv2.COLOR_BGR2RGB
                ),
                caption="Grad-CAM Heatmap",
                use_container_width=True
            )

        with col3:

            st.image(
                overlay_rgb,
                caption="Grad-CAM Overlay",
                use_container_width=True
            )

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        st.subheader(
            f"Predicted: "
            f"{GRADE_LABELS[pred_class]} "
            f"(Grade {pred_class}/4)"
        )

        st.write(
            f"**AI Confidence: "
            f"{confidence * 100:.1f}%**"
        )

        st.markdown(
            "### Clinical Explanation & Prevention Guidance"
        )

        st.markdown(
            generate_explanation(
                pred_class,
                heatmap_resized
            )
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

        st.info(
            "AI-assisted screening tool — clinical evaluation "
            "should be performed by a qualified healthcare professional."
        )

# =========================================================
# SCREENING HISTORY
# =========================================================

elif st.session_state.active_page == "Screening History":

    st.markdown(
        '<div class="main-title">Screening History</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.account_created:

        st.info(
            "Create a patient account to associate screenings "
            "with a patient."
        )

    else:

        st.success(
            f"Current patient: "
            f"{st.session_state.patient_name}"
        )

        st.write(
            "Persistent screening-history integration can be "
            "connected to your MySQL database here."
        )

# =========================================================
# MODEL INSIGHTS
# =========================================================

elif st.session_state.active_page == "Model Insights":

    st.markdown(
        '<div class="main-title">Model Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-subtitle">'
        'NETRA AI model and explainability pipeline'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Input Size",
            "224 × 224"
        )

    with c2:
        st.metric(
            "Classes",
            "5"
        )

    with c3:
        st.metric(
            "Explainability",
            "Grad-CAM"
        )

    st.markdown("---")

    st.write(
        """
        **Pipeline**

        Retinal Image → Preprocessing → AI Model →
        DR Grade → Grad-CAM → Clinical Explanation
        """
    )

# =========================================================
# ABOUT
# =========================================================

elif st.session_state.active_page == "About":

    st.markdown(
        '<div class="main-title">About NETRA AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        NETRA AI is an explainable AI-assisted retinal screening
        prototype designed to support diabetic retinopathy
        screening workflows.

        The system combines retinal image preprocessing,
        deep-learning based classification and Grad-CAM
        visual explanation.

        **NETRA AI is a screening-support prototype and does
        not replace professional ophthalmic diagnosis.**
        """
    )
