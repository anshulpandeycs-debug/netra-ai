import streamlit as st
import streamlit.components.v1 as components
import os

# Set page layout to wide and remove padding
st.set_page_config(
    page_title="NETRA AI — Explainable DR Screening",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inject CSS to make the HTML iframe fill the entire screen cleanly
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
            padding-left: 0rem !important;
            padding-right: 0rem !important;
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

# Read and display index.html
if os.path.exists("index.html"):
    with open("index.html", "r", encoding="utf-8") as f:
        html_code = f.read()
    components.html(html_code, height=1000, scrolling=True)
else:
    st.error("Error: `index.html` file not found in the root directory of your GitHub repository.")
