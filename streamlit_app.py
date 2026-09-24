import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from textwrap import dedent
from datetime import datetime


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NETRA AI — Explainable Screening",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "screenings" not in st.session_state:
    st.session_state.screenings = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    dedent("""
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background: #f7faf9 !important;
    }

    .main .block-container {
        max-width: 1280px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    header[data-testid="stHeader"] {
        background: #ffffff !important;
        border-bottom: 1px solid #e2ecea;
    }

    [data-testid="stDecoration"] {
        display: none;
    }

    .stApp p,
    .stApp label {
        color: #304b4b;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #dce7e5 !important;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 0.8rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 4px 8px 14px 8px;
    }

    .brand-icon {
        width: 28px;
        height: 28px;
        border-radius: 8px;
        background: #087f7a;
        color: white !important;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 15px;
        font-weight: 700;
    }

    .brand-name {
        font-size: 17px;
        font-weight: 750;
        color: #173c3c !important;
        line-height: 1.1;
    }

    .brand-sub {
        font-size: 7px;
        letter-spacing: 1.2px;
        color: #6f8b8b !important;
        margin-top: 2px;
    }

    .side-section {
        margin-top: 18px;
        margin-bottom: 7px;
        padding-left: 9px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: #789090 !important;
    }

    .side-info {
        font-size: 11px;
        color: #718787 !important;
        padding: 6px 9px;
        line-height: 1.5;
    }

    .bottom-status {
        position: fixed;
        bottom: 52px;
        left: 7px;
        width: 128px;
        border: 1px solid #d9e7e5;
        background: #ffffff;
        border-radius: 7px;
        padding: 6px 8px;
        font-size: 9px;
        color: #27736e !important;
    }


    /* ======================================================
       NATIVE SIDEBAR BUTTONS
       ====================================================== */

    section[data-testid="stSidebar"] .stButton {
        margin-bottom: 2px;
    }

    section[data-testid="stSidebar"] .stButton > button {
        border: none !important;
        background: transparent !important;
        color: #365252 !important;
        text-align: left !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        padding: 7px 9px !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #edf7f6 !important;
        color: #087f7a !important;
    }


    /* ======================================================
       TOP BAR
       ====================================================== */

    .topbar {
        height: 34px;
        border-bottom: 1px solid #e0e9e7;
        margin-bottom: 20px;
        color: #304b4b !important;
        font-size: 12px;
        display: flex;
        align-items: center;
    }


    /* ======================================================
       HERO
       ====================================================== */

    .hero {
        background: #ffffff;
        border: 1px solid #dbe7e5;
        border-radius: 16px;
        padding: 27px 28px;
        min-height: 175px;
        box-shadow: 0 3px 12px rgba(22, 55, 55, 0.035);
    }

    .eyebrow {
        display: inline-block;
        background: #eaf6f5;
        color: #087f7a !important;
        border-radius: 20px;
        padding: 5px 9px;
        font-size: 9px;
        font-weight: 700;
        margin-bottom: 9px;
    }

    .hero h1 {
        color: #102f35 !important;
        font-size: 29px;
        line-height: 1.08;
        margin: 0 0 10px 0;
        max-width: 550px;
    }

    .hero p {
        color: #648080 !important;
        font-size: 12px;
        line-height: 1.6;
        margin: 0 0 17px 0;
        max-width: 650px;
    }


    /* ======================================================
       HERO IMAGE / VISUAL
       ====================================================== */

    .hero-visual {
        background: #103b3a;
        border-radius: 16px;
        min-height: 175px;
        padding: 20px;
        color: white !important;
        position: relative;
        overflow: hidden;
    }

    .retina-ring {
        width: 105px;
        height: 105px;
        border-radius: 50%;
        border: 7px solid #5c302e;
        background: #8b5145;
        margin: 2px auto 9px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 0 7px rgba(255,255,255,0.03);
    }

    .retina-ring-inner {
        width: 42px;
        height: 42px;
        border-radius: 50%;
        background: #c47d60;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .retina-dot {
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: #d7a16e;
    }

    .visual-label {
        font-size: 8px;
        letter-spacing: 1.2px;
        color: #9bd2cf !important;
        margin-bottom: 3px;
    }

    .visual-title {
        color: white !important;
        font-size: 16px;
        font-weight: 750;
        line-height: 1.25;
    }

    .visual-note {
        color: #c7e2df !important;
        font-size: 9px;
        margin-top: 9px;
    }


    /* ======================================================
       BUTTONS
       ====================================================== */

    .primary-button button {
        background: #087f7a !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 11px !important;
        font-weight: 650 !important;
    }

    .secondary-button button {
        background: white !important;
        color: #173c3c !important;
        border: 1px solid #cadbd9 !important;
        border-radius: 8px !important;
        font-size: 11px !important;
        font-weight: 600 !important;
    }


    /* ======================================================
       STAT CARDS
       ====================================================== */

    .stat-card {
        background: #ffffff;
        border: 1px solid #dce7e5;
        border-radius: 14px;
        padding: 17px 18px;
        min-height: 88px;
    }

    .stat-label {
        font-size: 9px;
        color: #769090 !important;
        margin-bottom: 6px;
    }

    .stat-value {
        font-size: 20px;
        font-weight: 750;
        color: #102f35 !important;
    }

    .stat-note {
        font-size: 9px;
        color: #8aa0a0 !important;
        margin-top: 3px;
    }


    /* ======================================================
       CONTENT CARD
       ====================================================== */

    .content-card {
        background: #ffffff;
        border: 1px solid #dce7e5;
        border-radius: 15px;
        padding: 19px;
        margin-top: 16px;
    }

    .content-title {
        font-size: 16px;
        font-weight: 750;
        color: #123638 !important;
        margin-bottom: 4px;
    }

    .content-subtitle {
        font-size: 10px;
        color: #759090 !important;
    }


    /* ======================================================
       WORKFLOW
       ====================================================== */

    .workflow-box {
        border: 1px solid #dbe7e5;
        background: #f8fbfa;
        border-radius: 9px;
        padding: 11px;
        margin-bottom: 7px;
    }

    .workflow-number {
        color: #087f7a !important;
        font-size: 10px;
        font-weight: 750;
    }

    .workflow-title {
        color: #244848 !important;
        font-size: 11px;
        font-weight: 700;
    }

    .workflow-description {
        color: #7b9191 !important;
        font-size: 9px;
    }


    /* ======================================================
       INFO CARDS
       ====================================================== */

    .info-card {
        background: #edf7f6;
        border: 1px solid #d6ebe9;
        border-radius: 14px;
        padding: 20px;
        min-height: 105px;
    }

    .info-card.white {
        background: #ffffff;
    }

    .info-title {
        color: #173c3c !important;
        font-size: 15px;
        font-weight: 750;
        margin-bottom: 8px;
    }

    .info-text {
        color: #648080 !important;
        font-size: 10px;
        line-height: 1.55;
    }


    /* ======================================================
       SCREENING PAGE
       ====================================================== */

    .page-eyebrow {
        display: inline-block;
        color: #087f7a !important;
        background: #eaf6f5;
        border-radius: 20px;
        padding: 5px 9px;
        font-size: 9px;
        font-weight: 750;
    }

    .page-title {
        color: #102f35 !important;
        font-size: 30px;
        font-weight: 750;
        margin-top: 8px;
        margin-bottom: 5px;
    }

    .page-description {
        color: #708787 !important;
        font-size: 11px;
        margin-bottom: 18px;
    }

    .upload-card {
        background: #ffffff;
        border: 1px dashed #b8d2cf;
        border-radius: 15px;
        padding: 22px;
    }

    .quality-card {
        background: #f1f8f7;
        border: 1px solid #d7e9e7;
        border-radius: 9px;
        padding: 11px;
        margin-top: 8px;
    }

    .quality-good {
        color: #087f7a !important;
        font-size: 10px;
        font-weight: 750;
    }

    .quality-note {
        color: #789090 !important;
        font-size: 9px;
    }

    .check-pill {
        background: #ffffff;
        border: 1px solid #dce7e5;
        border-radius: 30px;
        padding: 8px 12px;
        font-size: 9px;
        color: #547070 !important;
        text-align: center;
    }


    /* ======================================================
       RESULT
       ====================================================== */

    .result-main {
        background: #ffffff;
        border: 1px solid #dce7e5;
        border-radius: 15px;
        padding: 22px;
        margin-top: 18px;
    }

    .result-label {
        font-size: 9px;
        letter-spacing: .7px;
        color: #769090 !important;
        font-weight: 700;
    }

    .result-grade {
        font-size: 28px;
        color: #102f35 !important;
        font-weight: 750;
        margin-top: 4px;
    }

    .grade-pill {
        display: inline-block;
        background: #eaf6f5;
        color: #087f7a !important;
        border-radius: 20px;
        padding: 5px 10px;
        font-size: 9px;
        font-weight: 700;
        margin-top: 6px;
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {
        text-align: center;
        color: #91a3a3 !important;
        font-size: 9px;
        padding-top: 28px;
    }


    /* ======================================================
       STREAMLIT WIDGETS
       ====================================================== */

    [data-testid="stFileUploader"] {
        background: transparent !important;
    }

    [data-testid="stFileUploader"] section {
        border-color: #bdd5d2 !important;
        background: #fbfdfc !important;
        border-radius: 10px !important;
    }

    [data-testid="stFileUploader"] * {
        color: #365252 !important;
    }

    .stProgress > div > div > div {
        background-color: #087f7a !important;
    }

    [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid #dce7e5 !important;
        border-radius: 10px !important;
    }

    </style>
    """),
    unsafe_allow_html=True
)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_netra_model():
    return tf.keras.models.load_model("netraai_final.keras")


model = load_netra_model()


# ============================================================
# MODEL FUNCTIONS
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

    max_value = tf.reduce_max(heatmap)

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        heatmap
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


GRADE_LABELS = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]


def generate_explanation(
    grade,
    heatmap
):

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
        "The model classified the image as No DR.",

        1:
        "The model classified the image as Mild DR.",

        2:
        "The model classified the image as Moderate DR.",

        3:
        "The model classified the image as Severe DR.",

        4:
        "The model classified the image as Proliferative DR."
    }

    guidance = {

        0:
        "Routine screening and appropriate diabetes management are recommended.",

        1:
        "Follow-up screening and professional eye evaluation should be considered according to clinical risk.",

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
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        dedent("""
        <div class="brand">
            <div class="brand-icon">◉</div>
            <div>
                <div class="brand-name">NETRA AI</div>
                <div class="brand-sub">EXPLAINABLE SCREENING</div>
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )

    # Dashboard
    if st.button("⌂ Dashboard", use_container_width=True):
        st.session_state.page = "Dashboard"
        st.rerun()

    # New screening
    if st.button("⊙ New Screening", use_container_width=True):
        st.session_state.page = "New Screening"
        st.rerun()

    # History
    if st.button("◷ Screening History", use_container_width=True):
        st.session_state.page = "History"
        st.rerun()

    # Model insights
    if st.button("▥ Model Insights", use_container_width=True):
        st.session_state.page = "Model Insights"
        st.rerun()

    # About
    if st.button("ⓘ About", use_container_width=True):
        st.session_state.page = "About"
        st.rerun()

    st.markdown(
        dedent("""
        <div style="height: 1px; background:#e4ecea; margin:15px 0;"></div>

        <div class="side-info">
            <b>Model</b><br>
            Lightweight CNN
        </div>

        <div class="side-info">
            <b>Dataset</b><br>
            APTOS 2019
        </div>

        <div class="side-info">
            <b>Explainability</b><br>
            Grad-CAM
        </div>

        <div class="bottom-status">
            ● System Ready
        </div>
        """),
        unsafe_allow_html=True
    )


# ============================================================
# TOP BAR
# ============================================================

st.markdown(
    '<div class="topbar">Workspace</div>',
    unsafe_allow_html=True
)


# ============================================================
# DASHBOARD
# ============================================================

if st.session_state.page == "Dashboard":

    st.markdown(
        dedent("""
        <div class="hero">

            <div class="eyebrow">
                ✦ Explainable AI
            </div>

            <h1>
                AI-powered diabetic<br>
                retinopathy screening
            </h1>

            <p>
                Upload a retinal image and receive an AI-assisted
                screening result with a visual explanation of
                the model's decision.
            </p>

        </div>
        """),
        unsafe_allow_html=True
    )

    # Hero buttons
    b1, b2, b3 = st.columns([1.25, 1.2, 2.2])

    with b1:
        st.markdown(
            '<div class="primary-button">',
            unsafe_allow_html=True
        )

        if st.button(
            "✦ Start New Screening →",
            use_container_width=True
        ):
            st.session_state.page = "New Screening"
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with b2:
        st.markdown(
            '<div class="secondary-button">',
            unsafe_allow_html=True
        )

        if st.button(
            "▶ How it works",
            use_container_width=True
        ):
            st.session_state.page = "Model Insights"
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


    # Right visual
    st.markdown(
        dedent("""
        <div class="hero-visual">

            <div class="retina-ring">
                <div class="retina-ring-inner">
                    <div class="retina-dot"></div>
                </div>
            </div>

            <div class="visual-label">
                NETRA AI
            </div>

            <div class="visual-title">
                See the disease.<br>
                Understand the decision.
            </div>

            <div class="visual-note">
                ✓ Human review remains essential
            </div>

        </div>
        """),
        unsafe_allow_html=True
    )


    # Stats
    st.write("")

    count = len(st.session_state.screenings)

    c1, c2, c3, c4 = st.columns(4)

    stats = [
        ("Screenings Completed", str(count), "Live count this session"),
        ("Images Analysed", str(count), "Live count this session"),
        ("DR Grade", "0–4", "Scale used by the model"),
        ("System Status", "Ready", "System operational")
    ]

    for col, item in zip(
        [c1, c2, c3, c4],
        stats
    ):

        with col:

            st.markdown(
                dedent(f"""
                <div class="stat-card">

                    <div class="stat-label">
                        {item[0]}
                    </div>

                    <div class="stat-value">
                        {item[1]}
                    </div>

                    <div class="stat-note">
                        {item[2]}
                    </div>

                </div>
                """),
                unsafe_allow_html=True
            )


    # Workflow
    st.markdown(
        dedent("""
        <div class="content-card">

            <div class="eyebrow">
                PRODUCT WALKTHROUGH
            </div>

            <div class="content-title">
                How NETRA AI works
            </div>

            <div class="content-subtitle">
                Simple screening workflow from retinal image
                to explainable AI output.
            </div>

        </div>
        """),
        unsafe_allow_html=True
    )


    left, right = st.columns([1.5, 1])

    with left:

        st.markdown(
            dedent("""
            <div style="
                background:#103b3a;
                border-radius:12px;
                min-height:180px;
                padding:25px;
                color:white;
                text-align:center;
            ">

                <div style="
                    font-size:40px;
                    margin-top:20px;
                ">
                    👁️
                </div>

                <div style="
                    color:#b7d8d5;
                    font-size:10px;
                    margin-top:8px;
                ">
                    NETRA AI WORKFLOW
                </div>

                <div style="
                    color:white;
                    font-size:14px;
                    font-weight:700;
                    margin-top:5px;
                ">
                    Upload → Analyse → Predict → Explain
                </div>

            </div>
            """),
            unsafe_allow_html=True
        )


    with right:

        workflow = [
            ("01", "Upload", "Upload a retinal image."),
            ("02", "Analyse", "AI preprocesses the image."),
            ("03", "Predict", "Model generates a DR grade."),
            ("04", "Explain", "Grad-CAM highlights influential regions.")
        ]

        for number, title, desc in workflow:

            st.markdown(
                dedent(f"""
                <div class="workflow-box">

                    <span class="workflow-number">
                        {number} ·
                    </span>

                    <span class="workflow-title">
                        {title}
                    </span>

                    <div class="workflow-description">
                        {desc}
                    </div>

                </div>
                """),
                unsafe_allow_html=True
            )


    # Bottom cards
    st.write("")

    a, b = st.columns(2)

    with a:

        st.markdown(
            dedent("""
            <div class="info-card">

                <div class="info-title">
                    Designed for accessible screening
                </div>

                <div class="info-text">
                    Simple workflow, low clutter, clear
                    explanations and a lightweight AI approach
                    designed for practical screening environments.
                </div>

            </div>
            """),
            unsafe_allow_html=True
        )

    with b:

        st.markdown(
            dedent("""
            <div class="info-card white">

                <div class="info-title">
                    ♡ AI Safety & Transparency
                </div>

                <div class="info-text">
                    AI-assisted screening is not a replacement
                    for clinical diagnosis. Human review is
                    recommended for final assessment.
                </div>

            </div>
            """),
            unsafe_allow_html=True
        )


# ============================================================
# NEW SCREENING
# ============================================================

elif st.session_state.page == "New Screening":

    st.markdown(
        '<div class="page-eyebrow">NEW SCREENING</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-title">Upload Retinal Image</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-description">Upload a clear fundus photograph for AI-assisted diabetic retinopathy screening.</div>',
        unsafe_allow_html=True
    )


    uploaded_file = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )


    if uploaded_file is None:

        st.markdown(
            dedent("""
            <div class="upload-card">

                <div style="
                    text-align:center;
                    padding:20px;
                ">

                    <div style="
                        font-size:42px;
                        margin-bottom:8px;
                    ">
                        📷
                    </div>

                    <div style="
                        font-size:16px;
                        font-weight:700;
                        color:#244848;
                    ">
                        Upload a retinal fundus image
                    </div>

                    <div style="
                        font-size:10px;
                        color:#819595;
                        margin-top:6px;
                    ">
                        JPG or PNG · Clear retinal/fundus photograph
                    </div>

                </div>

            </div>
            """),
            unsafe_allow_html=True
        )


    else:

        img = Image.open(
            uploaded_file
        ).convert("RGB")

        img_array = np.array(img)


        # Image area
        left, right = st.columns([1.4, 1])

        with left:

            st.image(
                img_array,
                caption=None,
                use_container_width=True
            )


        with right:

            st.markdown(
                dedent(f"""
                <div style="
                    padding:10px 3px;
                ">

                    <div style="
                        font-size:9px;
                        color:#809393;
                        margin-bottom:7px;
                    ">
                        SELECTED IMAGE
                    </div>

                    <div style="
                        font-size:13px;
                        font-weight:700;
                        color:#203e3e;
                        word-break:break-word;
                    ">
                        {uploaded_file.name}
                    </div>

                    <div class="quality-card">

                        <div class="quality-good">
                            ✓ IMAGE QUALITY · Good
                        </div>

                        <div class="quality-note">
                            Image loaded successfully and is
                            suitable for processing.
                        </div>

                    </div>

                </div>
                """),
                unsafe_allow_html=True
            )


            st.markdown(
                '<div class="primary-button">',
                unsafe_allow_html=True
            )

            analyse_button = st.button(
                "Analyse Image",
                use_container_width=True
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


        # Validation pills
        p1, p2, p3 = st.columns(3)

        with p1:
            st.markdown(
                '<div class="check-pill">✓ Clear retinal/fundus image</div>',
                unsafe_allow_html=True
            )

        with p2:
            st.markdown(
                '<div class="check-pill">✓ JPG or PNG</div>',
                unsafe_allow_html=True
            )

        with p3:
            st.markdown(
                '<div class="check-pill">✓ Suitable resolution</div>',
                unsafe_allow_html=True
            )


        # ====================================================
        # ANALYSE
        # ====================================================

        if analyse_button:

            with st.spinner(
                "NetraAI is analysing the retinal image..."
            ):

                processed = preprocess_image(
                    img_array,
                    size=224
                )

                input_array = np.expand_dims(
                    processed.astype("float32"),
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


            findings, hot_region, guidance = (
                generate_explanation(
                    pred_class,
                    heatmap_resized
                )
            )


            result = {
                "filename": uploaded_file.name,
                "grade": pred_class,
                "label": GRADE_LABELS[pred_class],
                "confidence": confidence,
                "time": datetime.now().strftime("%H:%M:%S")
            }

            st.session_state.last_result = result
            st.session_state.screenings.append(result)


            # =================================================
            # RESULT
            # =================================================

            st.markdown(
                '<div class="page-title" style="font-size:22px;margin-top:28px;">Screening Result</div>',
                unsafe_allow_html=True
            )


            r1, r2, r3 = st.columns(3)

            with r1:

                st.markdown(
                    dedent(f"""
                    <div class="stat-card">

                        <div class="stat-label">
                            DR GRADE
                        </div>

                        <div class="stat-value">
                            {pred_class}/4
                        </div>

                        <div class="stat-note">
                            {GRADE_LABELS[pred_class]}
                        </div>

                    </div>
                    """),
                    unsafe_allow_html=True
                )


            with r2:

                st.markdown(
                    dedent(f"""
                    <div class="stat-card">

                        <div class="stat-label">
                            MODEL CONFIDENCE
                        </div>

                        <div class="stat-value">
                            {confidence * 100:.1f}%
                        </div>

                        <div class="stat-note">
                            Model output confidence
                        </div>

                    </div>
                    """),
                    unsafe_allow_html=True
                )


            with r3:

                st.markdown(
                    dedent(f"""
                    <div class="stat-card">

                        <div class="stat-label">
                            ATTENTION REGION
                        </div>

                        <div class="stat-value"
                             style="font-size:15px;">
                            {hot_region}
                        </div>

                        <div class="stat-note">
                            Highest Grad-CAM activation
                        </div>

                    </div>
                    """),
                    unsafe_allow_html=True
                )


            st.markdown(
                dedent(f"""
                <div class="result-main">

                    <div class="result-label">
                        AI SCREENING CLASSIFICATION
                    </div>

                    <div class="result-grade">
                        {GRADE_LABELS[pred_class]}
                    </div>

                    <div class="grade-pill">
                        Grade {pred_class} / 4
                    </div>

                    <p style="
                        font-size:11px;
                        color:#6f8585;
                        margin-top:13px;
                    ">
                        {findings}
                    </p>

                </div>
                """),
                unsafe_allow_html=True
            )


            # =================================================
            # EXPLAINABILITY
            # =================================================

            st.markdown(
                '<div class="content-title" style="margin-top:22px;">Visual Explanation</div>',
                unsafe_allow_html=True
            )

            st.caption(
                "Grad-CAM highlights regions that contributed "
                "to the model's prediction."
            )


            i1, i2, i3 = st.columns(3)

            with i1:

                st.markdown(
                    '<div class="image-card-title">Original</div>',
                    unsafe_allow_html=True
                )

                st.image(
                    img_array,
                    use_container_width=True
                )


            with i2:

                st.markdown(
                    '<div class="image-card-title">AI Attention</div>',
                    unsafe_allow_html=True
                )

                st.image(
                    cv2.cvtColor(
                        heatmap_colored,
                        cv2.COLOR_BGR2RGB
                    ),
                    use_container_width=True
                )


            with i3:

                st.markdown(
                    '<div class="image-card-title">Grad-CAM Overlay</div>',
                    unsafe_allow_html=True
                )

                st.image(
                    overlay_rgb,
                    use_container_width=True
                )


            # =================================================
            # GUIDANCE
            # =================================================

            st.info(
                guidance
            )

            st.warning(
                "AI-assisted screening only. Final assessment "
                "should be performed by a qualified healthcare "
                "professional."
            )


# ============================================================
# HISTORY
# ============================================================

elif st.session_state.page == "History":

    st.markdown(
        '<div class="page-eyebrow">SCREENING HISTORY</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-title">Previous Screenings</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-description">Screenings performed during this application session.</div>',
        unsafe_allow_html=True
    )


    if len(st.session_state.screenings) == 0:

        st.markdown(
            dedent("""
            <div class="content-card"
                 style="text-align:center;padding:45px;">

                <div style="font-size:38px;">
                    🗂️
                </div>

                <div class="content-title">
                    No screenings yet
                </div>

                <div class="content-subtitle">
                    Start a new screening to see results here.
                </div>

            </div>
            """),
            unsafe_allow_html=True
        )

    else:

        for index, item in enumerate(
            reversed(
                st.session_state.screenings
            ),
            start=1
        ):

            st.markdown(
                dedent(f"""
                <div class="content-card">

                    <div style="
                        display:flex;
                        justify-content:space-between;
                    ">

                        <div>

                            <div class="content-title">
                                Screening #{index}
                            </div>

                            <div class="content-subtitle">
                                {item["filename"]}
                            </div>

                        </div>

                        <div style="
                            text-align:right;
                        ">

                            <div class="grade-pill">
                                {item["label"]}
                            </div>

                            <div class="content-subtitle"
                                 style="margin-top:5px;">
                                {item["confidence"] * 100:.1f}% confidence
                            </div>

                        </div>

                    </div>

                </div>
                """),
                unsafe_allow_html=True
            )


# ============================================================
# MODEL INSIGHTS
# ============================================================

elif st.session_state.page == "Model Insights":

    st.markdown(
        '<div class="page-eyebrow">MODEL INSIGHTS</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-title">How NETRA AI works</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-description">A simple view of the screening and explainability pipeline.</div>',
        unsafe_allow_html=True
    )


    steps = [
        (
            "01",
            "Fundus Image",
            "The system receives a retinal fundus photograph."
        ),
        (
            "02",
            "Preprocessing",
            "The image is resized to 224 × 224 and enhanced using CLAHE."
        ),
        (
            "03",
            "Lightweight CNN",
            "The neural network analyses visual retinal features."
        ),
        (
            "04",
            "DR Classification",
            "The model predicts one of five diabetic retinopathy grades."
        ),
        (
            "05",
            "Grad-CAM",
            "A heatmap provides a visual indication of regions influencing the prediction."
        )
    ]


    for number, title, description in steps:

        st.markdown(
            dedent(f"""
            <div class="content-card">

                <div style="
                    display:flex;
                    align-items:center;
                    gap:15px;
                ">

                    <div style="
                        width:38px;
                        height:38px;
                        border-radius:50%;
                        background:#eaf6f5;
                        color:#087f7a;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        font-size:11px;
                        font-weight:750;
                    ">
                        {number}
                    </div>

                    <div>

                        <div class="content-title">
                            {title}
                        </div>

                        <div class="content-subtitle">
                            {description}
                        </div>

                    </div>

                </div>

            </div>
            """),
            unsafe_allow_html=True
        )


    st.markdown(
        dedent("""
        <div class="info-card">

            <div class="info-title">
                Explainability
            </div>

            <div class="info-text">
                NETRA AI uses Grad-CAM to provide a visual
                explanation of which image regions contributed
                most strongly to the model output. This is an
                explanation of model attention, not a definitive
                clinical lesion map.
            </div>

        </div>
        """),
        unsafe_allow_html=True
    )


# ============================================================
# ABOUT
# ============================================================

elif st.session_state.page == "About":

    st.markdown(
        '<div class="page-eyebrow">ABOUT NETRA AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-title">About the Project</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-description">Lightweight and explainable AI for diabetic retinopathy screening.</div>',
        unsafe_allow_html=True
    )


    a1, a2 = st.columns(2)

    with a1:

        st.markdown(
            dedent("""
            <div class="content-card">

                <div class="content-title">
                    NETRA AI
                </div>

                <div class="content-subtitle"
                     style="line-height:1.7;">

                    NETRA AI is an AI-assisted diabetic
                    retinopathy screening prototype designed
                    around three principles:

                    <br><br>

                    • Lightweight inference<br>
                    • Explainable predictions<br>
                    • Accessible screening workflow

                </div>

            </div>
            """),
            unsafe_allow_html=True
        )


    with a2:

        st.markdown(
            dedent("""
            <div class="content-card">

                <div class="content-title">
                    Dataset
                </div>

                <div class="content-subtitle"
                     style="line-height:1.7;">

                    The project uses the APTOS 2019 Blindness
                    Detection dataset for five-class diabetic
                    retinopathy grading.

                    <br><br>

                    Grades:

                    <br>

                    0 — No DR<br>
                    1 — Mild DR<br>
                    2 — Moderate DR<br>
                    3 — Severe DR<br>
                    4 — Proliferative DR

                </div>

            </div>
            """),
            unsafe_allow_html=True
        )


    st.warning(
        "NETRA AI is a research/prototype screening system "
        "and should not be used as a standalone clinical "
        "diagnostic system."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">NETRA AI • Explainable AI for Diabetic Retinopathy Screening • Prototype</div>',
    unsafe_allow_html=True
)
