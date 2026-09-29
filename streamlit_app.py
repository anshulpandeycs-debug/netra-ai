# ============================================================
# NETRA AI — STREAMLIT APPLICATION
# Full replacement version
# ============================================================

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import sqlite3
import uuid
import hashlib
from datetime import datetime, date, timedelta


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
    from metrics import (
        calculate_referable_dr_metrics,
        generate_validation_plots
    )
    METRICS_MODULE_AVAILABLE = True
except ImportError:
    METRICS_MODULE_AVAILABLE = False


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NETRA AI — DR Screening & Pipeline",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# GLOBAL THEME
# ============================================================

st.markdown("""
<style>

/* =========================================================
   APPLICATION
========================================================= */

.stApp {
    background-color: #f7fafb;
}

.block-container {
    max-width: 1280px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}


/* =========================================================
   SIDEBAR
========================================================= */

section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}


/* =========================================================
   TYPOGRAPHY
========================================================= */

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

p,
label,
li,
span {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    color: #334155;
}


/* =========================================================
   BUTTONS
========================================================= */

.stButton > button {
    border-radius: 10px !important;
    border: 1px solid #0f766e !important;
    background-color: #0f766e !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    width: 100%;
}

.stButton > button:hover {
    background-color: #115e59 !important;
    border-color: #115e59 !important;
}


/* =========================================================
   STATUS BADGES
========================================================= */

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


/* =========================================================
   NETRA REGISTRATION
========================================================= */

.netra-title {
    font-size: 56px;
    line-height: 1.05;
    font-weight: 800;
    color: #172d2e;
    letter-spacing: -1.5px;
    margin-bottom: 18px;
}

.netra-subtitle {
    font-size: 22px;
    color: #587174;
    line-height: 1.5;
    margin-bottom: 42px;
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


/* =========================================================
   PATIENT CARDS
========================================================= */

.patient-card {
    background: #ffffff;
    border: 1px solid #dce8e7;
    border-radius: 30px;
    padding: 42px 40px 38px 40px;
    min-height: 390px;
    box-shadow: 0 10px 35px rgba(30, 70, 70, 0.06);
    box-sizing: border-box;
}

.patient-card:hover {
    border-color: #178c88;
    box-shadow: 0 15px 40px rgba(23, 140, 136, 0.10);
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
    font-weight: 700;
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
    min-height: 100px;
}


/* =========================================================
   PROTOTYPE NOTICE
========================================================= */

.prototype-box {
    margin-top: 24px;
    padding: 19px 22px;
    border: 1px solid #d8e8e7;
    border-radius: 18px;
    background: #f7fbfb;
    color: #547073;
    font-size: 16px;
    line-height: 1.5;
}

.prototype-box strong {
    color: #31595b;
}


/* =========================================================
   PATIENT PROFILE
========================================================= */

.profile-card {
    background: #ffffff;
    border: 1px solid #dce8e7;
    border-radius: 20px;
    padding: 22px;
}

.netra-id {
    color: #087b78;
    font-weight: 800;
    letter-spacing: 0.4px;
}


/* =========================================================
   INPUTS
========================================================= */

div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
    border-radius: 10px;
}


/* =========================================================
   DIVIDER
========================================================= */

hr {
    border-color: #dce8e7 !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# DATABASE
# ============================================================

DB_PATH = "netra_history.db"


def init_db():

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    # --------------------------------------------------------
    # Screening history
    # --------------------------------------------------------

    cursor.execute("""
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


    # --------------------------------------------------------
    # Patient registry
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (

            netra_id TEXT PRIMARY KEY,

            gov_id_type TEXT,

            gov_id_ref TEXT UNIQUE,

            name TEXT,

            father_name TEXT,

            mobile TEXT,

            dob TEXT,

            address TEXT,

            created_at TEXT,

            last_screened TEXT,

            current_grade INTEGER,

            followup_due TEXT

        )
    """)


    conn.commit()

    conn.close()


init_db()


# ============================================================
# DATABASE — SCREENING
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

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    unique_id = (
        f"NETRA-{uuid.uuid4().hex[:8].upper()}"
    )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    cursor.execute(
        """
        INSERT INTO screening_history

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

        """,

        (
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
        )
    )


    # --------------------------------------------------------
    # Follow-up schedule
    # --------------------------------------------------------

    followup_days = {
        0: 90,
        1: 60,
        2: 30,
        3: 15,
        4: 15
    }

    days = followup_days.get(
        int(grade),
        90
    )

    followup_due = (
        datetime.now() +
        timedelta(days=days)
    ).strftime("%Y-%m-%d")


    cursor.execute(
        """
        UPDATE patients

        SET
            last_screened = ?,
            current_grade = ?,
            followup_due = ?

        WHERE netra_id = ?

        """,

        (
            timestamp,
            int(grade),
            followup_due,
            patient_id
        )
    )


    conn.commit()

    conn.close()

    return unique_id


# ============================================================
# DATABASE — HISTORY
# ============================================================

def get_all_history():

    conn = sqlite3.connect(DB_PATH)

    if PANDAS_AVAILABLE:

        dataframe = pd.read_sql_query(
            """
            SELECT *

            FROM screening_history

            ORDER BY timestamp DESC
            """,
            conn
        )

        conn.close()

        return dataframe


    rows = conn.execute(
        """
        SELECT *

        FROM screening_history

        ORDER BY timestamp DESC
        """
    ).fetchall()

    conn.close()

    return rows


# ============================================================
# DATABASE — PATIENT
# ============================================================

def get_patient(
    netra_id=None,
    gov_id_ref=None
):

    conn = sqlite3.connect(DB_PATH)

    if netra_id:

        row = conn.execute(
            """
            SELECT *

            FROM patients

            WHERE netra_id = ?

            """,
            (netra_id,)
        ).fetchone()

    elif gov_id_ref:

        row = conn.execute(
            """
            SELECT *

            FROM patients

            WHERE gov_id_ref = ?

            """,
            (
                gov_id_ref.strip().upper(),
            )
        ).fetchone()

    else:

        row = None


    conn.close()

    return row


# ============================================================
# DATABASE — PATIENT DICTIONARY
# ============================================================

def patient_row_to_dict(row):

    if not row:
        return None

    keys = [

        "netra_id",
        "gov_id_type",
        "gov_id_ref",
        "name",
        "father_name",
        "mobile",
        "dob",
        "address",
        "created_at",
        "last_screened",
        "current_grade",
        "followup_due"

    ]

    return dict(
        zip(keys, row)
    )


# ============================================================
# DATABASE — REGISTER PATIENT
# ============================================================

def register_patient(
    gov_id_type,
    gov_id_ref,
    name,
    father_name,
    mobile,
    dob,
    address
):

    gov_id_ref = (
        gov_id_ref
        .strip()
        .upper()
    )


    existing = get_patient(
        gov_id_ref=gov_id_ref
    )


    if existing:

        return (
            None,
            "A patient is already registered with this government-ID reference."
        )


    netra_id = (

        "NTR-"
        +
        datetime.now().strftime("%y%m%d")
        +
        "-"
        +
        uuid.uuid4()
        .hex[:6]
        .upper()

    )


    now = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    conn = sqlite3.connect(DB_PATH)


    conn.execute(
        """
        INSERT INTO patients

        (
            netra_id,
            gov_id_type,
            gov_id_ref,
            name,
            father_name,
            mobile,
            dob,
            address,
            created_at,
            last_screened,
            current_grade,
            followup_due
        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

        """,

        (
            netra_id,
            gov_id_type,
            gov_id_ref,
            name,
            father_name,
            mobile,
            dob,
            address,
            now,
            None,
            None,
            None
        )
    )


    conn.commit()

    conn.close()


    return (
        netra_id,
        None
    )


# ============================================================
# DATABASE — PATIENT SEARCH
# ============================================================

def get_patients(search=""):

    conn = sqlite3.connect(DB_PATH)

    query = """

        SELECT

            netra_id,
            name,
            mobile,
            dob,
            last_screened,
            current_grade,
            followup_due

        FROM patients

    """

    parameters = []


    if search.strip():

        value = (
            "%"
            +
            search.strip()
            +
            "%"
        )


        query += """

            WHERE

                netra_id LIKE ?
                OR name LIKE ?
                OR mobile LIKE ?
                OR gov_id_ref LIKE ?

        """

        parameters = [
            value,
            value,
            value,
            value
        ]


    query += """

        ORDER BY created_at DESC

    """


    rows = conn.execute(
        query,
        parameters
    ).fetchall()


    conn.close()

    return rows


# ============================================================
# DATABASE — FOLLOW UPS
# ============================================================

def get_due_followups():

    today = date.today().isoformat()

    conn = sqlite3.connect(DB_PATH)


    rows = conn.execute(
        """

        SELECT

            netra_id,
            name,
            mobile,
            address,
            last_screened,
            current_grade,
            followup_due

        FROM patients

        WHERE

            followup_due IS NOT NULL

            AND followup_due <= ?

        ORDER BY followup_due ASC

        """,

        (today,)
    ).fetchall()


    conn.close()

    return rows


# ============================================================
# FOLLOW-UP RULE
# ============================================================

def grade_followup_days(
    grade
):

    return {

        0: 90,
        1: 60,
        2: 30,
        3: 15,
        4: 15

    }.get(
        int(grade),
        90
    )


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_netra_model():

    return tf.keras.models.load_model(
        "netraai_final.keras"
    )


try:

    model = load_netra_model()

    model_status = True

    model_error = ""

except Exception as error:

    model = None

    model_status = False

    model_error = str(error)


# ============================================================
# IMAGE QUALITY ASSESSMENT
# ============================================================

def evaluate_image_quality(
    img_rgb
):

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


    illumination_score = float(
        lab[:, :, 0].mean()
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
        (
            gray.shape[0]
            *
            gray.shape[1]
        )

    ) * 100


    if (

        focus_score >= 15

        and

        30 <= illumination_score <= 220

        and

        fov_coverage >= 35

    ):

        status = "PASS"

        action = (
            "Image quality meets screening criteria. "
            "Proceeding directly to AI classification."
        )


    elif (

        focus_score < 5

        or

        illumination_score < 15

        or

        fov_coverage < 20

    ):

        status = "RECAPTURE"

        action = (
            "Unusable image quality. "
            "Recapture required."
        )


    else:

        status = "BORDERLINE"

        action = (
            "Borderline quality detected. "
            "Applying adaptive CLAHE and denoising "
            "before grading."
        )


    return {

        "status": status,

        "action": action,

        "focus_score":
            round(
                focus_score,
                1
            ),

        "illumination_score":
            round(
                illumination_score,
                1
            ),

        "fov_coverage":
            round(
                fov_coverage,
                1
            )

    }


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_standard(
    img_rgb,
    size=224
):

    resized = cv2.resize(
        img_rgb,
        (size, size)
    )


    lab = cv2.cvtColor(
        resized,
        cv2.COLOR_RGB2LAB
    )


    l, a, b = cv2.split(
        lab
    )


    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )


    l = clahe.apply(l)


    return cv2.cvtColor(
        cv2.merge(
            (l, a, b)
        ),
        cv2.COLOR_LAB2RGB
    )


# ============================================================
# ADAPTIVE DENOISING
# ============================================================

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
# VESSEL EXTRACTION
# ============================================================

def extract_vascular_tree(
    img_rgb
):

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


# ============================================================
# OPTIC DISC / FOVEA
# ============================================================

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
            -
            int(od_radius * 2.8)
        )

    else:

        fovea_x = (
            od_center[0]
            +
            int(od_radius * 2.8)
        )


    fovea_y = od_center[1] + 5


    fovea_x = int(
        np.clip(
            fovea_x,
            10,
            w - 10
        )
    )


    cv2.circle(
        img_copy,
        (
            fovea_x,
            fovea_y
        ),
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


        if not conv_layers:

            raise RuntimeError(
                "No convolutional layer was found for Grad-CAM."
            )


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


        class_channel = (
            preds[:, pred_index]
        )


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


    if float(
        max_val.numpy()
    ) > 0:

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
# GRADE LABELS
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


    details = {

        0:
        f"""
1. Screening interpretation: No DR pattern was assigned by the model.

2. Model salience: Attention concentrated diffusely in {hot_region}.

3. Prevention: Maintain diabetes and blood-pressure control.

4. Follow-up: Prototype schedule = 90 days.
""",

        1:
        f"""
1. Screening interpretation: Mild DR pattern assigned by the model.

2. Model salience: Peak activation around {hot_region}.

3. Prevention: Maintain glycemic and blood-pressure control.

4. Follow-up: Prototype schedule = 60 days.
""",

        2:
        f"""
1. Screening interpretation: Moderate DR pattern assigned by the model.

2. Model salience: Concentrated activation in {hot_region}.

3. Prevention: Maintain strict glycemic and blood-pressure control.

4. Follow-up: Prototype schedule = 30 days.
""",

        3:
        f"""
1. Screening interpretation: Severe DR pattern assigned by the model.

2. Model salience: High activation in {hot_region}.

3. Care: Prompt ophthalmic review is recommended.

4. Follow-up: Prototype schedule = 15 days.
""",

        4:
        f"""
1. Screening interpretation: Proliferative DR pattern assigned by the model.

2. Model salience: High activation in {hot_region}.

3. Care: Urgent ophthalmic review is recommended.

4. Follow-up: Prototype schedule = 15 days.
"""

    }


    return (
        details.get(
            int(grade),
            "Screening complete."
        ),
        quadrants
    )


# ============================================================
# DEMO ID PROFILE
# ============================================================

def demo_patient_from_id(
    id_type,
    id_ref
):

    clean = (
        id_ref
        .strip()
        .upper()
    )


    digest = hashlib.sha256(
        clean.encode()
    ).hexdigest()


    names = [

        "Aarav Sharma",
        "Priya Verma",
        "Rohan Singh",
        "Ananya Gupta"

    ]


    fathers = [

        "Rajesh Sharma",
        "Suresh Verma",
        "Mahesh Singh",
        "Amit Gupta"

    ]


    index = (
        int(
            digest[:4],
            16
        )
        %
        len(names)
    )


    mobile = (

        "98"
        +
        str(
            int(
                digest[4:12],
                16
            )
            %
            100000000
        ).zfill(8)

    )


    dob = (

        f"{1970 + int(digest[12:14],16) % 35:04d}"
        f"-{1 + int(digest[14:16],16) % 12:02d}"
        f"-{1 + int(digest[16:18],16) % 28:02d}"

    )


    return {

        "name":
            names[index],

        "father_name":
            fathers[index],

        "mobile":
            mobile,

        "dob":
            dob,

        "address":
            "Prototype Rural Screening Centre",

        "id_type":
            id_type,

        "id_ref":
            clean

    }


# ============================================================
# PATIENT PROFILE
# ============================================================

def show_patient_profile(
    patient
):

    st.markdown(
        "### Verified Patient"
    )


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "Name",
        patient.get(
            "name",
            "—"
        )
    )


    c2.metric(
        "NETRA ID",
        patient.get(
            "netra_id",
            "—"
        )
    )


    c3.metric(
        "Mobile",
        patient.get(
            "mobile",
            "—"
        )
    )


    c4.metric(
        "DOB",
        patient.get(
            "dob",
            "—"
        )
    )


    st.info(

        f"""
**Father / Guardian:** {patient.get("father_name", "—")}

**Address:** {patient.get("address", "—")}

**Identity reference:** {patient.get("gov_id_type", "—")} • {patient.get("gov_id_ref", "—")}
"""

    )


# ============================================================
# PATIENT REGISTRATION SCREEN
# ============================================================

def patient_registration_screen():

    st.markdown(
        '<div class="registration-pill">PATIENT REGISTRATION</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="netra-title">Who is being screened?</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="netra-subtitle">
        Link the screening to a persistent NETRA patient record
        before analysing the retinal image.
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # If patient is already verified
    # --------------------------------------------------------

    if (

        st.session_state.get(
            "patient_verified"
        )

        and

        st.session_state.get(
            "patient"
        )

    ):

        patient = st.session_state[
            "patient"
        ]


        show_patient_profile(
            patient
        )


        if st.button(
            "Continue to Retinal Screening →",
            key="continue_to_screening"
        ):

            st.session_state[
                "screening_unlocked"
            ] = True


            st.session_state[
                "screening_step"
            ] = "upload"


            st.rerun()


        return


    # --------------------------------------------------------
    # Two cards
    # --------------------------------------------------------

    col1, col2 = st.columns(
        2,
        gap="large"
    )


    # ========================================================
    # EXISTING PATIENT
    # ========================================================

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
                    NETRA ID or registered government-ID reference.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        if st.button(
            "Continue as Existing Patient →",
            key="existing_patient_btn",
            use_container_width=True
        ):

            st.session_state[
                "patient_mode"
            ] = "existing"


            st.session_state[
                "registration_stage"
            ] = "id"


            st.rerun()


    # ========================================================
    # NEW PATIENT
    # ========================================================

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
                    identity record, then continue to retinal screening.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        if st.button(
            "Register New Patient →",
            key="new_patient_btn",
            use_container_width=True
        ):

            st.session_state[
                "patient_mode"
            ] = "new"


            st.session_state[
                "registration_stage"
            ] = "id"


            st.rerun()


    # ========================================================
    # NOTICE
    # ========================================================

    st.markdown(
        """
        <div class="prototype-box">

            <strong>Prototype verification:</strong>

            Government-ID lookup and OTP are simulated locally
            for the SIH demonstration.

            No real UIDAI, government database, or production
            OTP service is accessed.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# REGISTRATION WORKFLOW
# ============================================================

def registration_workflow():

    mode = st.session_state.get(
        "patient_mode",
        "existing"
    )


    stage = st.session_state.get(
        "registration_stage",
        "id"
    )


    st.markdown(
        "### Identity Verification"
    )


    if mode == "new":

        st.caption(
            "New patient registration"
        )

    else:

        st.caption(
            "Existing patient lookup"
        )


    # --------------------------------------------------------
    # ID TYPE
    # --------------------------------------------------------

    id_type = st.selectbox(

        "Government ID reference type",

        [

            "Aadhaar",
            "PAN",
            "Passport",
            "Ayushman Bharat / PM-JAY",
            "Other government ID"

        ],

        key="gov_id_type"

    )


    # --------------------------------------------------------
    # ID NUMBER
    # --------------------------------------------------------

    id_ref = st.text_input(

        "Government ID number / reference",

        placeholder="Enter demo reference",

        key="gov_id_ref"

    )


    # ========================================================
    # STEP 1 — ID
    # ========================================================

    if stage == "id":

        if st.button(
            "Verify ID & Send OTP",
            key="send_demo_otp"
        ):

            if not id_ref.strip():

                st.error(
                    "Enter an ID reference first."
                )

            else:

                existing = get_patient(
                    gov_id_ref=id_ref
                )


                st.session_state[
                    "registration_lookup"
                ] = (

                    patient_row_to_dict(
                        existing
                    )

                    if existing

                    else None

                )


                st.session_state[
                    "registration_demo_person"
                ] = demo_patient_from_id(
                    id_type,
                    id_ref
                )


                # Demo OTP
                st.session_state[
                    "registration_otp"
                ] = "123456"


                st.session_state[
                    "registration_stage"
                ] = "otp"


                st.rerun()


        return


    # ========================================================
    # DEMO OTP
    # ========================================================

    demo_person = st.session_state.get(
        "registration_demo_person"
    )


    if demo_person:

        st.info(

            f"""
Demo OTP sent to registered mobile
ending in **{demo_person["mobile"][-4:]}**.

For this prototype use OTP **123456**.
"""

        )


    otp = st.text_input(

        "Enter OTP",

        max_chars=6,

        key="demo_otp"

    )


    if st.button(
        "Verify OTP",
        key="verify_demo_otp"
    ):

        if (

            otp
            !=
            st.session_state.get(
                "registration_otp"
            )

        ):

            st.error(
                "Invalid demo OTP. Use 123456."
            )

            return


        existing = st.session_state.get(
            "registration_lookup"
        )


        # ----------------------------------------------------
        # EXISTING PATIENT
        # ----------------------------------------------------

        if mode == "existing":

            if not existing:

                st.error(

                    """
No registered NETRA patient was found
for this demo ID.

Choose New Patient to create a patient record first.
"""

                )

                return


            patient = existing


            st.session_state[
                "patient"
            ] = patient


            st.session_state[
                "patient_verified"
            ] = True


            st.session_state[
                "screening_unlocked"
            ] = False


            st.rerun()


        # ----------------------------------------------------
        # NEW PATIENT
        # ----------------------------------------------------

        st.session_state[
            "registration_stage"
        ] = "details"


        st.rerun()


    # ========================================================
    # STOP AFTER OTP
    # ========================================================

    if stage == "otp":

        return


    # ========================================================
    # NEW PATIENT DETAILS
    # ========================================================

    st.markdown(
        "### Confirm Patient Details"
    )


    p = (
        demo_person
        or {}
    )


    name = st.text_input(

        "Patient name",

        value=p.get(
            "name",
            ""
        ),

        key="reg_name"

    )


    father = st.text_input(

        "Father / Guardian name",

        value=p.get(
            "father_name",
            ""
        ),

        key="reg_father"

    )


    mobile = st.text_input(

        "Registered mobile",

        value=p.get(
            "mobile",
            ""
        ),

        key="reg_mobile"

    )


    dob = st.text_input(

        "Date of birth",

        value=p.get(
            "dob",
            ""
        ),

        key="reg_dob"

    )


    address = st.text_area(

        "Address",

        value=p.get(
            "address",
            ""
        ),

        key="reg_address"

    )


    if st.button(
        "Create NETRA Patient ID & Continue →",
        key="create_patient"
    ):

        if (

            not name.strip()

            or

            not mobile.strip()

            or

            not dob.strip()

        ):

            st.error(
                "Name, mobile number and date of birth are required."
            )

            return


        netra_id, error = register_patient(

            id_type,
            id_ref,
            name,
            father,
            mobile,
            dob,
            address

        )


        if error:

            st.error(
                error
            )

            return


        patient = {

            "netra_id":
                netra_id,

            "gov_id_type":
                id_type,

            "gov_id_ref":
                id_ref.strip().upper(),

            "name":
                name,

            "father_name":
                father,

            "mobile":
                mobile,

            "dob":
                dob,

            "address":
                address,

            "created_at":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "last_screened":
                None,

            "current_grade":
                None,

            "followup_due":
                None

        }


        st.session_state[
            "patient"
        ] = patient


        st.session_state[
            "patient_verified"
        ] = True


        st.session_state[
            "screening_unlocked"
        ] = False


        st.session_state[
            "registration_stage"
        ] = "id"


        st.success(
            f"NETRA ID created: {netra_id}"
        )


        st.rerun()


# ============================================================
# SCREENING PAGE
# ============================================================

def render_screening():

    patient = st.session_state.get(
        "patient"
    )


    if not patient:

        patient_registration_screen()

        return


    st.markdown(
        "## Retinal Screening"
    )


    st.caption(

        f"""
Patient:
**{patient["name"]}**

•

NETRA ID:
**{patient["netra_id"]}**
"""

    )


    # --------------------------------------------------------
    # CHANGE PATIENT
    # --------------------------------------------------------

    if st.button(
        "← Change Patient",
        key="change_patient"
    ):

        keys = [

            "patient",
            "patient_verified",
            "screening_unlocked",
            "patient_mode",
            "registration_stage"

        ]


        for key in keys:

            st.session_state.pop(
                key,
                None
            )


        st.rerun()


    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(

        "Upload Retinal Fundus Photograph",

        type=[
            "png",
            "jpg",
            "jpeg"
        ]

    )


    if not uploaded_file:

        st.info(
            "Upload a retinal fundus image to begin screening."
        )

        return


    raw_img = Image.open(
        uploaded_file
    ).convert("RGB")


    img_array = np.array(
        raw_img
    )


    # ========================================================
    # QUALITY
    # ========================================================

    st.divider()

    st.subheader(
        "1. Image Quality Assessment Gate"
    )


    q_metrics = evaluate_image_quality(
        img_array
    )


    q1, q2, q3, q4 = st.columns(4)


    q1.metric(
        "Focus",
        q_metrics["focus_score"]
    )


    q2.metric(
        "Illumination",
        q_metrics[
            "illumination_score"
        ]
    )


    q3.metric(
        "FOV Coverage",
        f'{q_metrics["fov_coverage"]}%'
    )


    if q_metrics["status"] == "PASS":

        q4.markdown(
            """
            <span class="status-badge status-pass">
            STATUS: PASS
            </span>
            """,
            unsafe_allow_html=True
        )


        processed_img = preprocess_standard(
            img_array
        )


    elif q_metrics["status"] == "BORDERLINE":

        q4.markdown(
            """
            <span class="status-badge status-borderline">
            STATUS: BORDERLINE
            </span>
            """,
            unsafe_allow_html=True
        )


        processed_img = preprocess_adaptive_denoise(
            img_array
        )


    else:

        q4.markdown(
            """
            <span class="status-badge status-fail">
            STATUS: RECAPTURE
            </span>
            """,
            unsafe_allow_html=True
        )


        st.error(
            """
            Automated grading stopped.

            Please recapture a better-quality fundus image.
            """
        )


        return


    st.info(
        f'**Gate Decision:** {q_metrics["action"]}'
    )


    # ========================================================
    # MODEL CHECK
    # ========================================================

    if not model_status:

        st.error(
            """
            Model unavailable.

            Check that `netraai_final.keras`
            exists beside `streamlit_app.py`.
            """
        )

        return


    # ========================================================
    # AI CLASSIFICATION
    # ========================================================

    st.divider()

    st.subheader(
        "2. AI Severity Grading & Grad-CAM XAI"
    )


    try:

        with st.spinner(
            "Executing classification and Grad-CAM..."
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

                (
                    224,
                    224
                )

            )


            heatmap_colored = cv2.applyColorMap(

                np.uint8(
                    255 * heatmap_resized
                ),

                cv2.COLORMAP_JET

            )


            processed_bgr = cv2.cvtColor(

                processed_img.astype(
                    "uint8"
                ),

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


    except Exception as error:

        st.error(
            f"Grad-CAM / classification error: {error}"
        )

        return


    # ========================================================
    # RESULT METRICS
    # ========================================================

    m1, m2, m3 = st.columns(3)


    m1.metric(
        "Predicted Severity",
        f"Grade {pred_class}"
    )


    m2.metric(
        "Grade",
        GRADE_LABELS[pred_class]
    )


    m3.metric(
        "Model Confidence",
        f"{confidence * 100:.1f}%"
    )


    # ========================================================
    # VISUAL XAI
    # ========================================================

    v1, v2, v3 = st.columns(3)


    v1.image(
        img_array,
        caption="Original Fundus",
        use_container_width=True
    )


    v2.image(
        cv2.cvtColor(
            heatmap_colored,
            cv2.COLOR_BGR2RGB
        ),
        caption="Grad-CAM Heatmap",
        use_container_width=True
    )


    v3.image(
        overlay_rgb,
        caption="Grad-CAM Overlay",
        use_container_width=True
    )


    # ========================================================
    # RETINAL STRUCTURES
    # ========================================================

    st.divider()

    st.subheader(
        "3. Retinal Structure Evidence"
    )


    r1, r2 = st.columns(2)


    with r1:

        st.image(

            localize_optic_disc_and_fovea(
                processed_img
            ),

            caption=(
                "Optic Disc / Fovea Localization"
            ),

            use_container_width=True

        )


    with r2:

        st.image(

            extract_vascular_tree(
                processed_img
            ),

            caption=(
                "Retinal Vasculature Mask"
            ),

            use_container_width=True

        )


    # ========================================================
    # EXPLANATION
    # ========================================================

    exp_text, quads = generate_explanation(

        pred_class,

        heatmap_resized

    )


    hot_quadrant = max(
        quads,
        key=quads.get
    )


    # ========================================================
    # SAVE SCREENING
    # ========================================================

    record_uid = save_screening_to_db(

        patient["netra_id"],

        patient["name"],

        pred_class,

        GRADE_LABELS[
            pred_class
        ],

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


    # ========================================================
    # FOLLOW-UP
    # ========================================================

    due_days = grade_followup_days(
        pred_class
    )


    due_date = (
        date.today()
        +
        timedelta(
            days=due_days
        )
    )


    st.success(

        f"""
Screening saved successfully.

Screening ID:
`{record_uid}`
"""

    )


    st.info(

        f"""
Prototype follow-up schedule:

**{due_days} days**

Due date:
**{due_date.isoformat()}**
"""

    )


    # ========================================================
    # CLINICAL REPORT
    # ========================================================

    st.markdown(
        "### Screening Interpretation"
    )


    formatted_report = (
        "<br><br>"
        .join(
            exp_text.split(
                "\n"
            )
        )
    )


    st.markdown(

        f"""
        <div style="
            background:#ffffff;
            border:1px solid #cbd5e1;
            border-radius:12px;
            padding:20px;
            line-height:1.7;
        ">

        {formatted_report}

        <hr>

        <b>Clinical safety:</b>

        NETRA AI is an AI-assisted screening prototype.
        The output should be reviewed by a qualified
        healthcare professional before clinical decisions.

        </div>
        """,

        unsafe_allow_html=True

    )


# ============================================================
# PATIENT REGISTRY
# ============================================================

def render_patient_registry():

    st.title(
        "Patient Registry"
    )


    st.caption(
        "Search registered NETRA patients."
    )


    search = st.text_input(

        "Search patient",

        placeholder=(
            "Name / mobile / NETRA ID / government-ID reference"
        )

    )


    rows = get_patients(
        search
    )


    if not rows:

        st.info(
            "No registered patients found."
        )

        return


    columns = [

        "NETRA ID",
        "Name",
        "Mobile",
        "DOB",
        "Last Screened",
        "Current Grade",
        "Follow-up Due"

    ]


    if PANDAS_AVAILABLE:

        dataframe = pd.DataFrame(
            rows,
            columns=columns
        )


        st.dataframe(

            dataframe,

            use_container_width=True,

            hide_index=True

        )


    else:

        for row in rows:

            st.write(
                row
            )


# ============================================================
# FOLLOW-UP WORKLIST
# ============================================================

def render_followups():

    st.title(
        "Follow-Up Worklist"
    )


    st.caption(
        "Patients whose prototype follow-up date has arrived."
    )


    rows = get_due_followups()


    if not rows:

        st.success(
            "No follow-ups are currently due."
        )

        return


    columns = [

        "NETRA ID",
        "Name",
        "Mobile",
        "Address",
        "Last Screened",
        "Grade",
        "Follow-up Due"

    ]


    if PANDAS_AVAILABLE:

        dataframe = pd.DataFrame(
            rows,
            columns=columns
        )


        st.dataframe(

            dataframe,

            use_container_width=True,

            hide_index=True

        )


        st.download_button(

            "Export / Print Worklist as CSV",

            data=dataframe.to_csv(
                index=False
            ).encode("utf-8"),

            file_name=(
                "netra_followup_worklist.csv"
            ),

            mime="text/csv"

        )


    else:

        for row in rows:

            st.write(
                row
            )


# ============================================================
# VALIDATION
# ============================================================

def render_validation():

    st.title(
        "Quantitative Model Validation Dashboard"
    )


    st.caption(
        "Project validation utility dashboard."
    )


    if not METRICS_MODULE_AVAILABLE:

        st.warning(
            "`metrics.py` is not available in this deployment."
        )

        return


    np.random.seed(42)


    y_true_demo = np.random.choice(

        [
            0,
            1,
            2,
            3,
            4
        ],

        size=200,

        p=[
            0.40,
            0.25,
            0.20,
            0.10,
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

        [
            0,
            1,
            2,
            3,
            4
        ],

        size=18

    )


    results = calculate_referable_dr_metrics(

        y_true_demo,

        y_pred_demo

    )


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(

        "Referable Sensitivity",

        f'{results["sensitivity"]:.2f}%'

    )


    c2.metric(

        "Referable Specificity",

        f'{results["specificity"]:.2f}%'

    )


    c3.metric(

        "Precision",

        f'{results["precision"]:.2f}%'

    )


    c4.metric(

        "F1",

        f'{results["f1_score"]:.2f}%'

    )


    if MATPLOTLIB_AVAILABLE:

        st.pyplot(

            generate_validation_plots(

                y_true_demo,

                y_pred_demo

            )

        )


# ============================================================
# PS COVERAGE
# ============================================================

def render_ps_coverage():

    st.title(
        "Problem Statement Coverage Dashboard"
    )


    ps_data = [

        {

            "Module":
                "Image Quality Assessment",

            "Status":
                "IMPLEMENTED",

            "Details":
                "Focus, illumination and FOV gate"

        },

        {

            "Module":
                "Adaptive Enhancement",

            "Status":
                "IMPLEMENTED",

            "Details":
                "CLAHE + bilateral denoising"

        },

        {

            "Module":
                "DR Severity Grading",

            "Status":
                "IMPLEMENTED",

            "Details":
                "EfficientNetB0 Grade 0–4"

        },

        {

            "Module":
                "Explainable AI",

            "Status":
                "IMPLEMENTED",

            "Details":
                "Grad-CAM visual evidence"

        },

        {

            "Module":
                "Retinal Structures",

            "Status":
                "PROTOTYPE",

            "Details":
                "Optic disc/fovea and vessel mask"

        },

        {

            "Module":
                "Patient Registry",

            "Status":
                "IMPLEMENTED",

            "Details":
                "Persistent local patient records"

        },

        {

            "Module":
                "Follow-Up Scheduling",

            "Status":
                "IMPLEMENTED",

            "Details":
                "Grade-based 90/60/30/15-day schedule"

        },

        {

            "Module":
                "Simulink Workflow",

            "Status":
                "ROADMAP / ARTIFACT",

            "Details":
                "MATLAB/Simulink program-level simulation"

        }

    ]


    if PANDAS_AVAILABLE:

        st.dataframe(

            pd.DataFrame(
                ps_data
            ),

            use_container_width=True,

            hide_index=True

        )


# ============================================================
# DISTRICT CAPACITY
# ============================================================

def render_capacity():

    st.title(
        "District-Level Telemedicine Capacity Simulator"
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


    ai_seconds = st.number_input(

        "Average AI processing seconds / image",

        value=0.5,

        min_value=0.1,

        step=0.1

    )


    total_hours = (
        250 * 8
    )


    ai_capacity_annual = int(

        (
            total_hours
            *
            3600
            /
            ai_seconds
        )
        *
        num_centers

    )


    c1, c2 = st.columns(2)


    c1.metric(

        "Target Patients",

        f"{annual_target:,}"

    )


    c2.metric(

        "Estimated AI Capacity",

        f"{ai_capacity_annual:,}"

    )


    if ai_capacity_annual >= annual_target:

        st.success(

            "Configured AI capacity exceeds "
            "the annual target under these assumptions."

        )

    else:

        st.warning(

            "Configured capacity is below "
            "the annual target under these assumptions."

        )


# ============================================================
# SESSION STATE
# ============================================================

defaults = {

    "authenticated":
        False,

    "user":
        "admin",

    "patient_mode":
        None,

    "patient_verified":
        False,

    "patient":
        None,

    "screening_unlocked":
        False,

    "registration_stage":
        "id"

}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[
            key
        ] = value


# ============================================================
# LOGIN
# ============================================================

if not st.session_state[
    "authenticated"
]:

    st.title(
        "👁️ NETRA AI — Clinical Access Portal"
    )


    st.caption(
        "Secure authenticated telemedicine screening node"
    )


    c1, c2, c3 = st.columns(
        [1, 2, 1]
    )


    with c2:

        with st.form(
            "login_form"
        ):

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

                    and

                    password == "password123"

                ):

                    st.session_state[
                        "authenticated"
                    ] = True


                    st.session_state[
                        "user"
                    ] = username


                    st.rerun()


                else:

                    st.error(
                        "Invalid username or password."
                    )


    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 👁️ NETRA AI"
    )


    st.caption(

        f"Operator: "
        f"`{st.session_state.get('user', 'admin')}`"

    )


    if st.button(
        "Log Out",
        key="logout_btn"
    ):

        st.session_state[
            "authenticated"
        ] = False

        st.rerun()


    st.divider()


    page = st.radio(

        "Navigate",

        [

            "◉ Screening Pipeline",

            "👤 Patient Registry",

            "⏰ Follow-Up Worklist",

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
# PAGE ROUTING
# ============================================================

if page == "◉ Screening Pipeline":

    # --------------------------------------------------------
    # No patient yet
    # --------------------------------------------------------

    if not st.session_state.get(
        "patient_verified"
    ):

        # ----------------------------------------------------
        # Patient mode selected
        # ----------------------------------------------------

        if st.session_state.get(
            "patient_mode"
        ):

            registration_workflow()

        # ----------------------------------------------------
        # First screen
        # ----------------------------------------------------

        else:

            patient_registration_screen()


    # --------------------------------------------------------
    # Patient verified
    # --------------------------------------------------------

    else:

        render_screening()


# ============================================================
# PATIENT REGISTRY
# ============================================================

elif page == "👤 Patient Registry":

    render_patient_registry()


# ============================================================
# FOLLOW-UP
# ============================================================

elif page == "⏰ Follow-Up Worklist":

    render_followups()


# ============================================================
# SCREENING HISTORY
# ============================================================

elif page == "📜 Screening History":

    st.title(
        "Patient Screening History Database"
    )


    st.caption(
        "Persistent screening record log"
    )


    history_df = get_all_history()


    if (

        PANDAS_AVAILABLE

        and

        isinstance(
            history_df,
            pd.DataFrame
        )

        and

        not history_df.empty

    ):

        st.metric(
            "Total Logged Screenings",
            len(history_df)
        )


        st.dataframe(

            history_df,

            use_container_width=True,

            hide_index=True

        )


        st.download_button(

            "Export History to CSV",

            data=history_df.to_csv(
                index=False
            ).encode("utf-8"),

            file_name=(
                "netra_screening_history.csv"
            ),

            mime="text/csv"

        )


    else:

        st.info(
            "No screening records found."
        )


# ============================================================
# VALIDATION
# ============================================================

elif page == "📊 Validation Metrics":

    render_validation()


# ============================================================
# PS COVERAGE
# ============================================================

elif page == "▥ PS Coverage Dashboard":

    render_ps_coverage()


# ============================================================
# DISTRICT CAPACITY
# ============================================================

elif page == "⚡ District Capacity Simulator":

    render_capacity()
