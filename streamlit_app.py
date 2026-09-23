import streamlit as st
import streamlit.components.v1 as components
import keras
import numpy as np
import cv2
import base64
import io
import os
from PIL import Image

st.set_page_config(
    page_title="NETRA AI — DR Screening",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide default Streamlit padding, header, and footer
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {
            padding: 0rem !important;
            max-width: 100% !important;
        }
        iframe {
            display: block;
            border: none;
            width: 100vw;
            height: 100vh;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_netra_model():
    model_path = "netraai_final.keras"
    if os.path.exists(model_path):
        try:
            return keras.models.load_model(model_path)
        except Exception:
            return None
    return None

model = load_netra_model()

# Render HTML UI
if os.path.exists("index.html"):
    with open("index.html", "r", encoding="utf-8") as f:
        html_code = f.read()
    components.html(html_code, height=950, scrolling=True)
else:
    st.error("Error: index.html file missing from repository root directory.")
