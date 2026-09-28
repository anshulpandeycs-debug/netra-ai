# ============================================================
# NETRA AI — PATIENT REGISTRATION SCREEN
# ============================================================

def patient_registration_screen():

    st.markdown("""
    <style>

    /* ---------- MAIN CONTAINER ---------- */

    .registration-wrapper {
        padding: 10px 0 30px 0;
    }

    /* ---------- SMALL PILL ---------- */

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

    /* ---------- MAIN HEADING ---------- */

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

    /* ---------- CARDS ---------- */

    .patient-card {
        background: #ffffff;
        border: 1px solid #dce8e7;
        border-radius: 30px;
        padding: 42px 40px 38px 40px;
        min-height: 390px;
        box-shadow: 0 10px 35px rgba(30, 70, 70, 0.06);
        transition: all 0.2s ease;
    }

    .patient-card:hover {
        border-color: #178c88;
        box-shadow: 0 15px 40px rgba(23, 140, 136, 0.10);
        transform: translateY(-2px);
    }

    /* ---------- ICON ---------- */

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

    /* ---------- CARD TEXT ---------- */

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

    /* ---------- INFO BOX ---------- */

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

    .prototype-box strong {
        color: #31595b;
    }

    /* ---------- STREAMLIT BUTTON ---------- */

    div.stButton > button {
        background: #087b78 !important;
        color: white !important;
        border: none !important;
        border-radius: 13px !important;
        padding: 15px 23px !important;
        font-size: 17px !important;
        font-weight: 800 !important;
        min-height: 52px !important;
        transition: all 0.2s ease !important;
    }

    div.stButton > button:hover {
        background: #066866 !important;
        transform: translateY(-1px);
        box-shadow: 0 7px 18px rgba(8,123,120,0.20);
    }

    </style>
    """, unsafe_allow_html=True)


    # ========================================================
    # HEADER
    # ========================================================

    st.markdown('<div class="registration-wrapper">', unsafe_allow_html=True)

    st.markdown(
        '<div class="registration-pill">PATIENT REGISTRATION</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="registration-title">Who is being screened?</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="registration-subtitle">'
        'Link the screening to a persistent NETRA patient record '
        'before analysing the retinal image.'
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # TWO PATIENT CARDS
    # ========================================================

    col1, col2 = st.columns(2, gap="large")


    # ========================================================
    # EXISTING PATIENT
    # ========================================================

    with col1:

        st.markdown("""
        <div class="patient-card">

            <div class="patient-icon">
                ⌕
            </div>

            <div class="patient-card-title">
                Existing Patient
            </div>

            <div class="patient-card-description">
                Find an existing NETRA patient using their
                NETRA ID or a registered government-ID reference.
            </div>

        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Continue as Existing Patient  →",
            key="existing_patient_btn",
            use_container_width=True
        ):
            st.session_state["patient_mode"] = "existing"
            st.session_state["screening_step"] = "verification"
            st.rerun()


    # ========================================================
    # NEW PATIENT
    # ========================================================

    with col2:

        st.markdown("""
        <div class="patient-card">

            <div class="patient-icon">
                +
            </div>

            <div class="patient-card-title">
                New Patient
            </div>

            <div class="patient-card-description">
                Create a NETRA ID, link the prototype identity
                record, then continue to the screening workflow.
            </div>

        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Register New Patient  →",
            key="new_patient_btn",
            use_container_width=True
        ):
            st.session_state["patient_mode"] = "new"
            st.session_state["screening_step"] = "registration"
            st.rerun()


    # ========================================================
    # PROTOTYPE NOTICE
    # ========================================================

    st.markdown("""
    <div class="prototype-box">

        <strong>Prototype verification:</strong>
        government-ID verification is simulated locally for the
        SIH prototype. No real UIDAI/government API or OTP service
        is accessed by this demo.

    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
