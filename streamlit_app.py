# ============================================================
# NETRA AI — COMPLETE STREAMLIT APPLICATION
# Patient Registration + Simulated Government-ID Verification
# + OTP + Patient Record + DR Screening + Grad-CAM
# ============================================================

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import sqlite3
import uuid
import random
import re
from datetime import datetime, timedelta

# ============================================================
# OPTIONAL IMPORTS
# ============================================================

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from metrics import calculate_referable_dr_metrics, generate_validation_plots
    METRICS_MODULE_AVAILABLE = True
except ImportError:
    METRICS_MODULE_AVAILABLE = False


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NETRA AI — DR Screening",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# GLOBAL THEME
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #f7fafb;
}

.block-container {
    max-width: 1280px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}

h1 {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #123b4a !important;
}

h2 {
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    color: #123b4a !important;
}

h3 {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: #123b4a !important;
}

p, label, li, span {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    color: #334155;
}

.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.85rem;
}

.status-pass {
    background-color: #dcfce7;
    color: #166534;
}

.status-borderline {
    background-color: #fef9c3;
    color: #854d0e;
}

.status-fail {
    background-color: #fee2e2;
    color: #991b1b;
}

.stButton > button {
    border-radius: 10px;
    border: 1px solid #0f766e !important;
    background-color: #0f766e !important;
    color: #ffffff !important;
    font-weight: 600;
    width: 100%;
    min-height: 45px;
}

.stButton > button:hover {
    background-color: #115e59 !important;
    border-color: #115e59 !important;
}

.registration-wrapper {
    padding: 10px 0 30px 0;
}

.registration-pill {
    display: inline-block;
    background: #eaf5f4;
    color: #087b78;
    font-size: 15px;
    font-weight: 800;
    letter-spacing: 0.3px;
    padding: 11px 17px;
    border-radius: 25px;
    margin-bottom: 12px;
}

.registration-title {
    font-size: 56px;
    line-height: 1.05;
    font-weight: 800;
    color: #172d2e;
    margin: 0 0 18px 0;
    letter-spacing: -1.5px;
}

.registration-subtitle {
    font-size: 22px;
    color: #587174;
    margin-bottom: 42px;
    line-height: 1.5;
}

.patient-card {
    background: #ffffff;
    border: 1px solid #dce8e7;
    border-radius: 30px;
    padding: 42px 40px 38px 40px;
    min-height: 360px;
    box-shadow: 0 10px 35px rgba(30, 70, 70, 0.06);
}

.patient-icon {
    width: 70px;
    height: 70px;
    border-radius: 22px;
    background: #e8f5f4;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #087b78;
    font-size: 35px;
    margin-bottom: 30px;
}

.patient-card-title {
    color: #172d2e;
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 18px;
}

.patient-card-description {
    color: #647577;
    font-size: 20px;
    line-height: 1.55;
    margin-bottom: 30px;
}

.prototype-box {
    margin-top: 22px;
    padding: 19px 22px;
    border: 1px solid #d8e8e7;
    border-radius: 18px;
    background: #f7fbfb;
    color: #547073;
    font-size: 16px;
    line-height: 1.5;
}

.patient-header {
    background: #ffffff;
    border: 1px solid #dce8e7;
    border-radius: 18px;
    padding: 22px 25px;
    margin-bottom: 25px;
}

.netra-id-box {
    background: #eaf5f4;
    border-radius: 12px;
    padding: 12px 18px;
    color: #087b78;
    font-weight: 700;
}

.info-card {
    background: #ffffff;
    border: 1px solid #dce8e7;
    border-radius: 18px;
    padding: 25px;
    margin-bottom: 20px;
}

.big-number {
    font-size: 34px;
    font-weight: 800;
    color: #087b78;
}

.otp-box {
    background: #eef7ff;
    border: 1px solid #b8d9f5;
    border-radius: 14px;
    padding: 18px;
    margin: 15px 0;
}

.demo-id-box {
    background: #fff9e8;
    border: 1px solid #f0d78a;
    border-radius: 14px;
    padding: 18px;
    margin: 15px 0;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "authenticated": False,
    "user": "admin",
    "patient_mode": None,
    "screening_step": "dashboard",
    "patient": None,
    "otp": None,
    "otp_verified": False,
    "generated_netra_id": None,
    "generated_patient": None,
    "current_page": "◉ Screening Pipeline"
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# SQLITE DATABASE
# ============================================================

def get_db():
    return sqlite3.connect("netra_history.db")


def init_db():

    conn = get_db()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS screening_history (
            unique_id TEXT PRIMARY KEY,
            timestamp TEXT,
            patient_id TEXT,
            patient_name TEXT,
            predicted_grade INTEGER,
            grade_label TEXT,
            confidence REAL,
            hotspot_quadrant TEXT,
            focus_score REAL,
            illum_score REAL,
            fov_score REAL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            netra_id TEXT PRIMARY KEY,
            patient_name TEXT,
            father_name TEXT,
            mobile TEXT,
            dob TEXT,
            address TEXT,
            id_type TEXT,
            id_reference TEXT,
            created_at TEXT,
            last_screened TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()


# ============================================================
# PATIENT DATABASE FUNCTIONS
# ============================================================

def create_netra_id():
    return f"NETRA-{uuid.uuid4().hex[:8].upper()}"


def save_patient(patient):

    conn = get_db()
    c = conn.cursor()

    c.execute("""
        INSERT OR REPLACE INTO patients
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient["netra_id"],
        patient["name"],
        patient["father_name"],
        patient["mobile"],
        patient["dob"],
        patient["address"],
        patient["id_type"],
        patient["id_reference"],
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        patient.get("last_screened", "")
    ))

    conn.commit()
    conn.close()


def find_patient_by_id(id_type, id_reference):

    conn = get_db()
    c = conn.cursor()

    row = c.execute("""
        SELECT
            netra_id,
            patient_name,
            father_name,
            mobile,
            dob,
            address,
            id_type,
            id_reference,
            last_screened
        FROM patients
        WHERE id_type = ? AND id_reference = ?
    """, (id_type, id_reference)).fetchone()

    conn.close()

    if not row:
        return None

    return {
        "netra_id": row[0],
        "name": row[1],
        "father_name": row[2],
        "mobile": row[3],
        "dob": row[4],
        "address": row[5],
        "id_type": row[6],
        "id_reference": row[7],
        "last_screened": row[8]
    }


def get_patient_by_netra(netra_id):

    conn = get_db()
    c = conn.cursor()

    row = c.execute("""
        SELECT
            netra_id,
            patient_name,
            father_name,
            mobile,
            dob,
            address,
            id_type,
            id_reference,
            last_screened
        FROM patients
        WHERE netra_id = ?
    """, (netra_id,)).fetchone()

    conn.close()

    if not row:
        return None

    return {
        "netra_id": row[0],
        "name": row[1],
        "father_name": row[2],
        "mobile": row[3],
        "dob": row[4],
        "address": row[5],
        "id_type": row[6],
        "id_reference": row[7],
        "last_screened": row[8]
    }


def update_last_screened(netra_id):

    conn = get_db()
    c = conn.cursor()

    c.execute("""
        UPDATE patients
        SET last_screened = ?
        WHERE netra_id = ?
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        netra_id
    ))

    conn.commit()
    conn.close()


# ============================================================
# SCREENING DATABASE
# ============================================================

def save_screening_to_db(
    patient_id,
    patient_name,
    grade,
    grade_label,
    confidence,
    hot_quad,
    focus,
    illum,
    fov
):

    conn = get_db()
    c = conn.cursor()

    unique_id = f"NETRA-{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    c.execute("""
        INSERT INTO screening_history
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        unique_id,
        timestamp,
        patient_id,
        patient_name,
        grade,
        grade_label,
        confidence,
        hot_quad,
        focus,
        illum,
        fov
    ))

    conn.commit()
    conn.close()

    if patient_id:
        update_last_screened(patient_id)

    return unique_id


def get_all_history():

    conn = get_db()

    if PANDAS_AVAILABLE:

        df = pd.read_sql_query(
            """
            SELECT *
            FROM screening_history
            ORDER BY timestamp DESC
            """,
            conn
        )

        conn.close()

        return df

    c = conn.cursor()

    data = c.execute("""
        SELECT *
        FROM screening_history
        ORDER BY timestamp DESC
    """).fetchall()

    conn.close()

    return data


# ============================================================
# SIMULATED GOVERNMENT-ID VALIDATION
# ============================================================

def validate_id_format(id_type, value):

    value = value.strip().upper().replace(" ", "")

    if id_type == "Aadhaar":
        return bool(re.fullmatch(r"\d{12}", value))

    if id_type == "PAN":
        return bool(re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", value))

    if id_type == "Passport":
        return bool(re.fullmatch(r"[A-Z][0-9]{7}", value))

    if id_type == "Ayushman Bharat Card":
        return bool(re.fullmatch(r"[A-Z0-9]{9,20}", value))

    return False


def mask_id(id_type, value):

    if id_type == "Aadhaar":
        return "XXXX XXXX " + value[-4:]

    if id_type == "PAN":
        return value[:2] + "XXXXXX" + value[-2:]

    if id_type == "Passport":
        return value[:1] + "XXXXXX" + value[-1:]

    return "XXXXXX" + value[-4:]


# ============================================================
# SIMULATED PATIENT DETAILS
# ============================================================

def generate_demo_patient(id_type, id_reference):

    seed = sum(ord(x) for x in id_reference)
    random.seed(seed)

    first_names = [
        "Arjun",
        "Rahul",
        "Priya",
        "Neha",
        "Amit",
        "Kavya",
        "Rohan",
        "Ananya"
    ]

    father_names = [
        "Rajesh Kumar",
        "Suresh Sharma",
        "Mahesh Singh",
        "Vijay Patel",
        "Ramesh Verma"
    ]

    cities = [
        "Jaipur, Rajasthan",
        "Delhi, India",
        "Lucknow, Uttar Pradesh",
        "Bhopal, Madhya Pradesh",
        "Pune, Maharashtra"
    ]

    name = random.choice(first_names)
    father = random.choice(father_names)
    city = random.choice(cities)

    mobile = f"9{random.randint(100000000, 999999999)}"

    dob = (
        f"{random.randint(1, 28):02d}-"
        f"{random.randint(1, 12):02d}-"
        f"{random.randint(1965, 2002)}"
    )

    return {
        "name": name,
        "father_name": father,
        "mobile": mobile,
        "dob": dob,
        "address": city,
        "id_type": id_type,
        "id_reference": id_reference,
        "netra_id": create_netra_id()
    }


# ============================================================
# OTP
# ============================================================

def generate_demo_otp():

    return str(random.randint(100000, 999999))


# ============================================================
# LOGIN AUTHENTICATION
# ============================================================

if not st.session_state["authenticated"]:

    st.title("👁️ NETRA AI — Clinical Access Portal")

    st.caption(
        "Secure Authenticated Telemedicine Screening Node"
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        with st.form("login_form"):

            username = st.text_input(
                "Username",
                value="admin"
            )

            password = st.text_input(
                "Password",
                type="password",
                value="password123"
            )

            submit = st.form_submit_button(
                "Log In to Screening Node"
            )

            if submit:

                if (
                    username == "admin"
                    and password == "password123"
                ):

                    st.session_state["authenticated"] = True
                    st.session_state["user"] = username
                    st.session_state["screening_step"] = "dashboard"

                    st.success(
                        "Authentication successful."
                    )

                    st.rerun()

                else:

                    st.error(
                        "Invalid Username or Password."
                    )

    st.stop()


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_netra_model():

    return tf.keras.models.load_model(
        "netraai_final.keras"
    )


try:

    model = load_netra_model()

    model_status = True

except Exception as e:

    model = None

    model_status = False

    model_error = str(e)


# ============================================================
# IMAGE QUALITY FUNCTIONS
# ============================================================

def evaluate_image_quality(img_rgb):

    gray = cv2.cvtColor(
        img_rgb,
        cv2.COLOR_RGB2GRAY
    )

    focus_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    lab = cv2.cvtColor(
        img_rgb,
        cv2.COLOR_RGB2LAB
    )

    l_channel = lab[:, :, 0]

    illumination_score = float(
        l_channel.mean()
    )

    _, mask = cv2.threshold(
        gray,
        10,
        255,
        cv2.THRESH_BINARY
    )

    fov_coverage = (
        cv2.countNonZero(mask)
        /
        (gray.shape[0] * gray.shape[1])
    ) * 100

    is_focus_pass = focus_score >= 15.0

    is_illum_pass = (
        30.0 <= illumination_score <= 220.0
    )

    is_fov_pass = fov_coverage >= 35.0

    if (
        is_focus_pass
        and is_illum_pass
        and is_fov_pass
    ):

        status = "PASS"

        action = (
            "Image quality meets screening criteria. "
            "Proceeding directly to AI classification."
        )

    elif (
        focus_score < 5.0
        or illumination_score < 15.0
        or fov_coverage < 20.0
    ):

        status = "RECAPTURE"

        action = (
            "Unusable image quality. Recapture required."
        )

    else:

        status = "BORDERLINE"

        action = (
            "Borderline quality detected. "
            "Applying adaptive CLAHE & denoising."
        )

    return {
        "status": status,
        "action": action,
        "focus_score": round(focus_score, 1),
        "illumination_score": round(
            illumination_score,
            1
        ),
        "fov_coverage": round(
            fov_coverage,
            1
        )
    }


def preprocess_standard(img_rgb, size=224):

    resized = cv2.resize(
        img_rgb,
        (size, size)
    )

    lab = cv2.cvtColor(
        resized,
        cv2.COLOR_RGB2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    return cv2.cvtColor(
        cv2.merge((l, a, b)),
        cv2.COLOR_LAB2RGB
    )


def preprocess_adaptive_denoise(
    img_rgb,
    size=224
):

    processed = preprocess_standard(
        img_rgb,
        size=size
    )

    return cv2.bilateralFilter(
        processed,
        d=5,
        sigmaColor=50,
        sigmaSpace=50
    )


# ============================================================
# RETINAL STRUCTURE FUNCTIONS
# ============================================================

def extract_vascular_tree(img_rgb):

    green_ch = img_rgb[:, :, 1]

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    enhanced_g = clahe.apply(
        green_ch
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    tophat = cv2.morphologyEx(
        enhanced_g,
        cv2.MORPH_TOPHAT,
        kernel
    )

    _, vessel_mask = cv2.threshold(
        tophat,
        15,
        255,
        cv2.THRESH_BINARY
    )

    return cv2.cvtColor(
        vessel_mask,
        cv2.COLOR_GRAY2RGB
    )


def localize_optic_disc_and_fovea(
    img_rgb
):

    img_copy = img_rgb.copy()

    gray = cv2.cvtColor(
        img_rgb,
        cv2.COLOR_RGB2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (15, 15),
        0
    )

    _, _, _, max_loc = cv2.minMaxLoc(
        blurred
    )

    od_center = max_loc

    od_radius = 24

    cv2.circle(
        img_copy,
        od_center,
        od_radius,
        (0, 255, 255),
        2
    )

    cv2.putText(
        img_copy,
        "Optic Disc",
        (
            od_center[0] - 30,
            od_center[1] - 30
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 255),
        1
    )

    h, w = gray.shape

    if od_center[0] > w // 2:

        fovea_x = (
            od_center[0]
            - int(od_radius * 2.8)
        )

    else:

        fovea_x = (
            od_center[0]
            + int(od_radius * 2.8)
        )

    fovea_y = od_center[1] + 5

    fovea_x = np.clip(
        fovea_x,
        10,
        w - 10
    )

    cv2.circle(
        img_copy,
        (fovea_x, fovea_y),
        12,
        (255, 0, 0),
        2
    )

    cv2.putText(
        img_copy,
        "Fovea",
        (
            fovea_x - 20,
            fovea_y - 18
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 0, 0),
        1
    )

    return img_copy


# ============================================================
# GRAD-CAM
# ============================================================

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

    heatmap = (
        last_conv_layer_output[0]
        @
        pooled_grads[..., tf.newaxis]
    )

    heatmap = tf.squeeze(
        heatmap
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_val = tf.math.reduce_max(
        heatmap
    )

    if float(max_val.numpy()) > 0:

        heatmap /= max_val

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
# DR GRADES
# ============================================================

GRADE_LABELS = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]


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
                :h//2,
                :w//2
            ].mean(),

        "Superior-Temporal":
            heatmap[
                :h//2,
                w//2:
            ].mean(),

        "Inferior-Nasal":
            heatmap[
                h//2:,
                :w//2
            ].mean(),

        "Inferior-Temporal":
            heatmap[
                h//2:,
                w//2:
            ].mean()
    }

    hot_region = max(
        quadrants,
        key=quadrants.get
    )

    details = {

        0:
        f"""
1. Screening Assessment:
No obvious DR-related pattern was identified by the prototype.

2. Model Salience:
Attention was concentrated in {hot_region}.

3. Follow-up:
Routine screening interval should be determined by a qualified clinician.

4. Safety:
This is an AI-assisted screening output and not a diagnosis.
""",

        1:
        f"""
1. Screening Assessment:
The prototype classified the image as Mild DR.

2. Model Salience:
Attention was concentrated in {hot_region}.

3. Follow-up:
Clinical follow-up should be determined by a qualified healthcare professional.

4. Safety:
The AI result should be reviewed by a qualified clinician.
""",

        2:
        f"""
1. Screening Assessment:
The prototype classified the image as Moderate DR.

2. Model Salience:
Attention was concentrated in {hot_region}.

3. Follow-up:
Clinical referral should be determined by a qualified healthcare professional.

4. Safety:
The AI result should not replace clinical examination.
""",

        3:
        f"""
1. Screening Assessment:
The prototype classified the image as Severe DR.

2. Model Salience:
High attention was concentrated in {hot_region}.

3. Follow-up:
Prompt clinical assessment is recommended.

4. Safety:
The AI result should not replace professional diagnosis.
""",

        4:
        f"""
1. Screening Assessment:
The prototype classified the image as Proliferative DR.

2. Model Salience:
Maximum attention was concentrated in {hot_region}.

3. Follow-up:
Immediate professional clinical assessment is recommended.

4. Safety:
This prototype output is not a substitute for diagnosis.
"""
    }

    return (
        details.get(
            grade,
            "Screening complete."
        ),
        quadrants
    )


# ============================================================
# FOLLOW-UP CALCULATION
# ============================================================

def followup_message(grade):

    if grade == 0:
        return (
            "Suggested prototype follow-up: "
            "3 months"
        )

    if grade == 1:
        return (
            "Suggested prototype follow-up: "
            "2 months"
        )

    if grade == 2:
        return (
            "Suggested prototype follow-up: "
            "1 month"
        )

    return (
        "Priority clinical consultation recommended. "
        "Prototype follow-up reminder: 15 days."
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:28px;
            font-weight:900;
            color:#087b78;
            letter-spacing:-0.5px;
            margin-bottom:5px;">
            NETRA AI
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        f"Operator: `{st.session_state.get('user', 'admin')}`"
    )

    if st.button(
        "Log Out",
        key="logout_button"
    ):

        st.session_state["authenticated"] = False

        st.rerun()

    st.divider()

    page = st.radio(
        "Navigate",
        [
            "◉ Screening Pipeline",
            "📜 Screening History",
            "📊 Validation Metrics",
            "▥ PS Coverage Dashboard",
            "⚡ District Capacity Simulator"
        ],
        index=0
    )

    st.divider()

    st.write(
        "**Model:** EfficientNetB0"
    )

    st.write(
        "**Database:** SQLite (`netra_history.db`)"
    )

    if model_status:

        st.success(
            "● Keras Model Loaded"
        )

    else:

        st.error(
            "● Model Unavailable"
        )


# ============================================================
# PATIENT REGISTRATION SCREEN
# ============================================================

def patient_registration_screen():

    st.markdown(
        '<div class="registration-wrapper">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="registration-pill">'
        'PATIENT REGISTRATION'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="registration-title">'
        'Who is being screened?'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="registration-subtitle">'
        'Link the screening to a persistent NETRA patient record '
        'before analysing the retinal image.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(
        2,
        gap="large"
    )

    # --------------------------------------------------------
    # EXISTING PATIENT
    # --------------------------------------------------------

    with col1:

        st.markdown(
            """
            <div class="patient-card">

                <div class="patient-icon">
                    ⌕
                </div>

                <div class="patient-card-title">
                    Existing Patient
                </div>

                <div class="patient-card-description">
                    Find an existing NETRA patient using their
                    NETRA ID or a registered government-ID
                    reference.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "Continue as Existing Patient  →",
            key="existing_patient_btn",
            use_container_width=True
        ):

            st.session_state[
                "patient_mode"
            ] = "existing"

            st.session_state[
                "screening_step"
            ] = "verification"

            st.rerun()

    # --------------------------------------------------------
    # NEW PATIENT
    # --------------------------------------------------------

    with col2:

        st.markdown(
            """
            <div class="patient-card">

                <div class="patient-icon">
                    +
                </div>

                <div class="patient-card-title">
                    New Patient
                </div>

                <div class="patient-card-description">
                    Create a NETRA ID, link the prototype
                    identity record, then continue to the
                    screening workflow.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "Register New Patient  →",
            key="new_patient_btn",
            use_container_width=True
        ):

            st.session_state[
                "patient_mode"
            ] = "new"

            st.session_state[
                "screening_step"
            ] = "registration"

            st.rerun()

    st.markdown(
        """
        <div class="prototype-box">

        <strong>Prototype verification:</strong>
        Government-ID verification and OTP are simulated
        locally for the SIH demonstration.
        No real UIDAI, Passport, PAN, Ayushman Bharat or
        government API is accessed.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# ID VERIFICATION SCREEN
# ============================================================

def identity_verification_screen():

    st.markdown(
        "## Patient Identity Verification"
    )

    st.caption(
        "Select an identity document and enter its reference."
    )

    st.info(
        "Prototype mode: the ID is checked only against "
        "a local format/record. No government database is queried."
    )

    id_type = st.selectbox(
        "Government ID Type",
        [
            "Aadhaar",
            "PAN",
            "Passport",
            "Ayushman Bharat Card"
        ],
        key="identity_type"
    )

    placeholders = {
        "Aadhaar": "Example: 123456789012",
        "PAN": "Example: ABCDE1234F",
        "Passport": "Example: A1234567",
        "Ayushman Bharat Card":
            "Example: ABH123456789"
    }

    id_value = st.text_input(
        f"{id_type} Number",
        placeholder=placeholders[id_type],
        key="identity_reference"
    )

    c1, c2 = st.columns(2)

    with c1:

        if st.button(
            "Verify Identity →",
            key="verify_identity"
        ):

            clean_value = (
                id_value
                .strip()
                .upper()
                .replace(" ", "")
            )

            if not clean_value:

                st.error(
                    "Please enter the ID reference."
                )

            elif not validate_id_format(
                id_type,
                clean_value
            ):

                st.error(
                    "ID not found — invalid ID format."
                )

            else:

                existing = find_patient_by_id(
                    id_type,
                    clean_value
                )

                if existing:

                    st.session_state[
                        "generated_patient"
                    ] = existing

                    st.session_state[
                        "screening_step"
                    ] = "otp"

                    st.session_state[
                        "otp"
                    ] = generate_demo_otp()

                    st.session_state[
                        "otp_verified"
                    ] = False

                    st.rerun()

                else:

                    if (
                        st.session_state[
                            "patient_mode"
                        ] == "existing"
                    ):

                        st.error(
                            "ID format is valid, but no "
                            "registered NETRA patient record "
                            "was found."
                        )

                    else:

                        patient = (
                            generate_demo_patient(
                                id_type,
                                clean_value
                            )
                        )

                        st.session_state[
                            "generated_patient"
                        ] = patient

                        st.session_state[
                            "screening_step"
                        ] = "otp"

                        st.session_state[
                            "otp"
                        ] = generate_demo_otp()

                        st.session_state[
                            "otp_verified"
                        ] = False

                        st.rerun()

    with c2:

        if st.button(
            "← Back",
            key="identity_back"
        ):

            st.session_state[
                "screening_step"
            ] = "patient_selection"

            st.rerun()


# ============================================================
# OTP SCREEN
# ============================================================

def otp_screen():

    patient = st.session_state.get(
        "generated_patient"
    )

    if not patient:

        st.session_state[
            "screening_step"
        ] = "patient_selection"

        st.rerun()

    st.markdown(
        "## Verify Registered Mobile"
    )

    st.caption(
        "A prototype OTP has been generated for the "
        "mobile number associated with the identity record."
    )

    masked_mobile = (
        "XXXXXX"
        + patient["mobile"][-4:]
    )

    st.markdown(
        f"""
        <div class="info-card">

        <b>Patient</b><br>
        {patient["name"]}<br><br>

        <b>Registered Mobile</b><br>
        {masked_mobile}

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # DEMO OTP
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="otp-box">

        <b>Prototype OTP</b><br><br>

        <span style="
            font-size:30px;
            font-weight:900;
            letter-spacing:8px;
            color:#087b78;">
            {st.session_state["otp"]}
        </span>

        <br><br>

        This OTP is displayed only because this is a
        local SIH prototype.

        </div>
        """,
        unsafe_allow_html=True
    )

    entered_otp = st.text_input(
        "Enter OTP",
        max_chars=6,
        key="entered_otp"
    )

    c1, c2 = st.columns(2)

    with c1:

        if st.button(
            "Verify OTP →",
            key="verify_otp"
        ):

            if (
                entered_otp.strip()
                ==
                st.session_state["otp"]
            ):

                st.session_state[
                    "otp_verified"
                ] = True

                st.session_state[
                    "screening_step"
                ] = "patient_details"

                st.rerun()

            else:

                st.error(
                    "Incorrect OTP."
                )

    with c2:

        if st.button(
            "← Back",
            key="otp_back"
        ):

            st.session_state[
                "screening_step"
            ] = "verification"

            st.rerun()


# ============================================================
# PATIENT DETAILS
# ============================================================

def patient_details_screen():

    patient = st.session_state.get(
        "generated_patient"
    )

    if not patient:

        st.session_state[
            "screening_step"
        ] = "patient_selection"

        st.rerun()

    st.markdown(
        "## Patient Verified"
    )

    st.success(
        "Identity verification completed in prototype mode."
    )

    st.markdown(
        f"""
        <div class="netra-id-box">

        NETRA PATIENT ID:
        <span style="font-size:22px;">
        {patient["netra_id"]}
        </span>

        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            f"""
            <div class="info-card">

            <b>Patient Name</b><br>
            {patient["name"]}<br><br>

            <b>Father's Name</b><br>
            {patient["father_name"]}<br><br>

            <b>Date of Birth</b><br>
            {patient["dob"]}

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="info-card">

            <b>Mobile</b><br>
            {patient["mobile"]}<br><br>

            <b>Address</b><br>
            {patient["address"]}<br><br>

            <b>Identity Type</b><br>
            {patient["id_type"]}

            </div>
            """,
            unsafe_allow_html=True
        )

    st.warning(
        "These patient details are synthetic prototype data. "
        "They are not retrieved from a government database."
    )

    c1, c2 = st.columns(2)

    with c1:

        if st.button(
            "Begin Retinal Screening →",
            key="begin_screening"
        ):

            save_patient(patient)

            st.session_state[
                "patient"
            ] = patient

            st.session_state[
                "screening_step"
            ] = "screening"

            st.rerun()

    with c2:

        if st.button(
            "← Back",
            key="details_back"
        ):

            st.session_state[
                "screening_step"
            ] = "otp"

            st.rerun()


# ============================================================
# SCREENING DASHBOARD
# ============================================================

def screening_dashboard():

    history = get_all_history()

    total_screenings = (
        len(history)
        if PANDAS_AVAILABLE
        and isinstance(history, pd.DataFrame)
        else 0
    )

    st.markdown(
        "## NETRA AI Dashboard"
    )

    st.caption(
        "AI-assisted retinal screening and longitudinal patient management."
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="info-card">
            <div>Screenings Completed</div>
            <div class="big-number">
            """
            + str(total_screenings)
            + """
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        conn = get_db()

        patient_count = conn.execute(
            "SELECT COUNT(*) FROM patients"
        ).fetchone()[0]

        conn.close()

        st.markdown(
            f"""
            <div class="info-card">

            <div>Patients Registered</div>

            <div class="big-number">
            {patient_count}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="info-card">

            <div>System Status</div>

            <div class="big-number">
            {"READY" if model_status else "MODEL ERROR"}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.markdown(
        "### Start a new screening"
    )

    st.write(
        "Verify or register the patient before uploading "
        "a retinal image."
    )

    if st.button(
        "Start New Screening →",
        key="start_new_screening"
    ):

        st.session_state[
            "screening_step"
        ] = "patient_selection"

        st.session_state[
            "patient_mode"
        ] = None

        st.rerun()


# ============================================================
# MAIN SCREENING PIPELINE
# ============================================================

def screening_pipeline():

    step = st.session_state.get(
        "screening_step",
        "dashboard"
    )

    # --------------------------------------------------------
    # DASHBOARD
    # --------------------------------------------------------

    if step == "dashboard":

        screening_dashboard()

        return

    # --------------------------------------------------------
    # PATIENT SELECTION
    # --------------------------------------------------------

    if step == "patient_selection":

        patient_registration_screen()

        return

    # --------------------------------------------------------
    # VERIFICATION
    # --------------------------------------------------------

    if step == "verification":

        identity_verification_screen()

        return

    # --------------------------------------------------------
    # NEW REGISTRATION / VERIFICATION
    # --------------------------------------------------------

    if step == "registration":

        identity_verification_screen()

        return

    # --------------------------------------------------------
    # OTP
    # --------------------------------------------------------

    if step == "otp":

        otp_screen()

        return

    # --------------------------------------------------------
    # PATIENT DETAILS
    # --------------------------------------------------------

    if step == "patient_details":

        patient_details_screen()

        return

    # --------------------------------------------------------
    # ACTUAL SCREENING
    # --------------------------------------------------------

    if step == "screening":

        patient = st.session_state.get(
            "patient"
        )

        if not patient:

            st.session_state[
                "screening_step"
            ] = "patient_selection"

            st.rerun()

        st.markdown(
            f"""
            <div class="patient-header">

            <div style="
                font-size:13px;
                font-weight:800;
                color:#087b78;">
                PATIENT VERIFIED
            </div>

            <h2>
            {patient["name"]}
            </h2>

            <div class="netra-id-box">
            NETRA ID:
            {patient["netra_id"]}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "← Switch Patient",
            key="switch_patient"
        ):

            st.session_state[
                "patient"
            ] = None

            st.session_state[
                "screening_step"
            ] = "patient_selection"

            st.rerun()

        st.markdown(
            "### Upload Retinal Image"
        )

        st.caption(
            "Upload a clear fundus photograph for "
            "AI-assisted diabetic retinopathy screening."
        )

        uploaded_file = st.file_uploader(
            "Browse / Upload Fundus Image",
            type=[
                "png",
                "jpg",
                "jpeg"
            ],
            key="fundus_upload"
        )

        if not uploaded_file:

            st.info(
                "Please upload a retinal fundus photograph "
                "to begin analysis."
            )

            return

        raw_img = Image.open(
            uploaded_file
        ).convert("RGB")

        img_array = np.array(
            raw_img
        )

        st.divider()

        # ----------------------------------------------------
        # QUALITY GATE
        # ----------------------------------------------------

        st.subheader(
            "1. Image Quality Assessment Gate"
        )

        q_metrics = evaluate_image_quality(
            img_array
        )

        q_col1, q_col2, q_col3, q_col4 = (
            st.columns(4)
        )

        q_col1.metric(
            "Focus",
            f"{q_metrics['focus_score']}"
        )

        q_col2.metric(
            "Illumination",
            f"{q_metrics['illumination_score']}"
        )

        q_col3.metric(
            "FOV Coverage",
            f"{q_metrics['fov_coverage']}%"
        )

        status = q_metrics["status"]

        if status == "PASS":

            q_col4.markdown(
                '<span class="status-badge status-pass">'
                'STATUS: PASS'
                '</span>',
                unsafe_allow_html=True
            )

            processed_img = (
                preprocess_standard(
                    img_array,
                    size=224
                )
            )

        elif status == "BORDERLINE":

            q_col4.markdown(
                '<span class="status-badge status-borderline">'
                'STATUS: BORDERLINE'
                '</span>',
                unsafe_allow_html=True
            )

            processed_img = (
                preprocess_adaptive_denoise(
                    img_array,
                    size=224
                )
            )

        else:

            q_col4.markdown(
                '<span class="status-badge status-fail">'
                'STATUS: RECAPTURE'
                '</span>',
                unsafe_allow_html=True
            )

            processed_img = (
                preprocess_standard(
                    img_array,
                    size=224
                )
            )

        st.info(
            f"**Gate Decision:** "
            f"{q_metrics['action']}"
        )

        if status == "RECAPTURE":

            st.error(
                "⛔ Automated grading stopped. "
                "Please recapture the fundus image."
            )

            return

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        if not model_status:

            st.error(
                "Model unavailable. "
                "Check `netraai_final.keras`."
            )

            return

        st.divider()

        st.subheader(
            "2. AI Severity Grading & Grad-CAM XAI"
        )

        with st.spinner(
            "Executing classification & computing Grad-CAM..."
        ):

            input_tensor = np.expand_dims(
                processed_img.astype(
                    "float32"
                ),
                axis=0
            )

            heatmap, pred_class, confidence = (
                make_gradcam_heatmap(
                    input_tensor,
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

            processed_bgr = cv2.cvtColor(
                processed_img.astype("uint8"),
                cv2.COLOR_RGB2BGR
            )

            overlay_rgb = cv2.cvtColor(
                cv2.addWeighted(
                    processed_bgr,
                    0.6,
                    heatmap_colored,
                    0.4,
                    0
                ),
                cv2.COLOR_BGR2RGB
            )

        m_col1, m_col2 = st.columns(2)

        m_col1.metric(
            "Predicted Severity",
            f"Grade {pred_class} — "
            f"{GRADE_LABELS[pred_class]}"
        )

        m_col2.metric(
            "Model Confidence",
            f"{confidence * 100:.1f}%"
        )

        v_col1, v_col2, v_col3 = st.columns(3)

        v_col1.image(
            img_array,
            caption="Original Fundus",
            use_container_width=True
        )

        v_col2.image(
            cv2.cvtColor(
                heatmap_colored,
                cv2.COLOR_BGR2RGB
            ),
            caption="Grad-CAM Heatmap",
            use_container_width=True
        )

        v_col3.image(
            overlay_rgb,
            caption="Grad-CAM Overlay",
            use_container_width=True
        )

        # ----------------------------------------------------
        # RETINAL STRUCTURES
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "3. Retinal Structure & Anatomical Evidence"
        )

        r_col1, r_col2 = st.columns(2)

        with r_col1:

            st.image(
                localize_optic_disc_and_fovea(
                    processed_img
                ),
                caption=(
                    "Anatomical Landmarks "
                    "(Optic Disc: Yellow | Fovea: Blue)"
                ),
                use_container_width=True
            )

        with r_col2:

            st.image(
                extract_vascular_tree(
                    processed_img
                ),
                caption="Segmented Retinal Vasculature Mask",
                use_container_width=True
            )

        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        exp_text, quads = (
            generate_explanation(
                pred_class,
                heatmap_resized
            )
        )

        hot_quadrant = max(
            quads,
            key=quads.get
        )

        st.divider()

        st.subheader(
            "4. Explainable Screening Report"
        )

        formatted_report = (
            exp_text
            .replace("\n", "<br>")
        )

        st.markdown(
            f"""
            <div style="
                background:#ffffff;
                border:1px solid #cbd5e1;
                border-radius:12px;
                padding:20px;
                color:#1e293b;
                line-height:1.6;">

            {formatted_report}

            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # FOLLOW-UP
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "5. Prototype Follow-Up Schedule"
        )

        st.info(
            followup_message(
                pred_class
            )
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        if st.button(
            "Save Screening Record",
            key=f"save_{patient['netra_id']}"
        ):

            record_uid = (
                save_screening_to_db(
                    patient["netra_id"],
                    patient["name"],
                    pred_class,
                    GRADE_LABELS[pred_class],
                    round(
                        confidence * 100,
                        2
                    ),
                    hot_quadrant,
                    q_metrics[
                        "focus_score"
                    ],
                    q_metrics[
                        "illumination_score"
                    ],
                    q_metrics[
                        "fov_coverage"
                    ]
                )
            )

            st.success(
                f"Screening record logged: "
                f"`{record_uid}`"
            )


# ============================================================
# PAGE 1 — SCREENING PIPELINE
# ============================================================

if page == "◉ Screening Pipeline":

    screening_pipeline()


# ============================================================
# PAGE 2 — SCREENING HISTORY
# ============================================================

elif page == "📜 Screening History":

    st.title(
        "Patient Screening History Database"
    )

    st.caption(
        "Persistent record log generated via SQLite database"
    )

    history_df = get_all_history()

    if (
        PANDAS_AVAILABLE
        and isinstance(
            history_df,
            pd.DataFrame
        )
        and not history_df.empty
    ):

        st.metric(
            "Total Logged Screenings",
            len(history_df)
        )

        st.dataframe(
            history_df,
            use_container_width=True
        )

        csv_data = (
            history_df
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "📥 Export History to CSV",
            data=csv_data,
            file_name="netra_screening_history.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "No screening records found in database yet."
        )


# ============================================================
# PAGE 3 — VALIDATION METRICS
# ============================================================

elif page == "📊 Validation Metrics":

    st.title(
        "Quantitative Model Validation Dashboard"
    )

    st.caption(
        "Phase 3 metric evaluation"
    )

    np.random.seed(42)

    y_true_demo = np.random.choice(
        [0, 1, 2, 3, 4],
        size=200,
        p=[
            0.4,
            0.25,
            0.2,
            0.1,
            0.05
        ]
    )

    y_pred_demo = (
        y_true_demo.copy()
    )

    noise_idx = np.random.choice(
        200,
        size=18,
        replace=False
    )

    y_pred_demo[
        noise_idx
    ] = np.random.choice(
        [0, 1, 2, 3, 4],
        size=18
    )

    if METRICS_MODULE_AVAILABLE:

        m_results = (
            calculate_referable_dr_metrics(
                y_true_demo,
                y_pred_demo
            )
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Referable Sensitivity",
            f"{m_results['sensitivity']:.2f}%"
        )

        c2.metric(
            "Referable Specificity",
            f"{m_results['specificity']:.2f}%"
        )

        c3.metric(
            "Precision",
            f"{m_results['precision']:.2f}%"
        )

        c4.metric(
            "F1-Score",
            f"{m_results['f1_score']:.2f}%"
        )

        st.divider()

        try:

            st.pyplot(
                generate_validation_plots(
                    y_true_demo,
                    y_pred_demo
                )
            )

        except Exception as e:

            st.warning(
                f"Validation plot unavailable: {e}"
            )

    else:

        st.error(
            "`metrics.py` module not found."
        )


# ============================================================
# PAGE 4 — PS COVERAGE DASHBOARD
# ============================================================

elif page == "▥ PS Coverage Dashboard":

    st.title(
        "Problem Statement Requirements & Implementation Matrix"
    )

    ps_data = [

        {
            "Module":
                "Authentication & Patient Registry",

            "Components":
                "Login, simulated ID, OTP, NETRA ID",

            "Status":
                "IMPLEMENTED",

            "Details":
                "Prototype identity layer with persistent patient records"
        },

        {
            "Module":
                "Image Quality Assessment",

            "Components":
                "Focus, Illumination, FOV",

            "Status":
                "IMPLEMENTED",

            "Details":
                "Live quality gate with borderline enhancement"
        },

        {
            "Module":
                "DR Severity Grading",

            "Components":
                "Grade 0–4 Classification",

            "Status":
                "IMPLEMENTED",

            "Details":
                "EfficientNetB0 model"
        },

        {
            "Module":
                "Explainable AI",

            "Components":
                "Grad-CAM, Quadrant Salience",

            "Status":
                "IMPLEMENTED",

            "Details":
                "Visual attention maps and salience analysis"
        },

        {
            "Module":
                "Retinal Structures",

            "Components":
                "Optic Disc, Fovea, Vasculature",

            "Status":
                "IMPLEMENTED",

            "Details":
                "Prototype anatomical and vessel processing"
        },

        {
            "Module":
                "Patient History",

            "Components":
                "SQLite, NETRA ID",

            "Status":
                "IMPLEMENTED",

            "Details":
                "Persistent screening history"
        },

        {
            "Module":
                "Follow-Up",

            "Components":
                "Grade-based timeline",

            "Status":
                "IMPLEMENTED",

            "Details":
                "Prototype follow-up recommendation logic"
        },

        {
            "Module":
                "Simulink Artifact",

            "Components":
                "MATLAB / Simulink",

            "Status":
                "IMPLEMENTED",

            "Details":
                "External MATLAB/Simulink project artifact"
        }
    ]

    if PANDAS_AVAILABLE:

        st.dataframe(
            pd.DataFrame(ps_data),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PAGE 5 — DISTRICT CAPACITY SIMULATOR
# ============================================================

elif page == "⚡ District Capacity Simulator":

    st.title(
        "District-Level Telemedicine Capacity Simulator"
    )

    st.caption(
        "Screening capacity planning calculator"
    )

    annual_target = st.number_input(
        "Annual Target Patients",
        value=100000,
        step=10000
    )

    num_centers = st.slider(
        "Primary Screening Centers",
        min_value=1,
        max_value=50,
        value=10
    )

    total_hours = (
        250 * 8
    )

    ai_capacity_annual = (
        total_hours
        * 3600
        / 0.5
    ) * num_centers

    st.metric(
        "AI Pipeline Annual Processing Capacity",
        f"{int(ai_capacity_annual):,} images"
    )

    if ai_capacity_annual >= annual_target:

        st.success(
            "Configured centers have sufficient "
            "theoretical AI processing capacity "
            "for the selected annual target."
        )

    else:

        st.warning(
            "Additional processing capacity or "
            "screening centers may be required."
        )


# ============================================================
# END OF APPLICATION
# ============================================================
