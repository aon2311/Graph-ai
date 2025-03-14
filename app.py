import streamlit as st
from home import display_home
from ml_model import display_ml_models
from nn_model import display_nn_model
from demo_prediction import display_demo_prediction
import os
from pathlib import Path
def local_css(file_name):
    file_path = Path(__file__).parent / file_name  # หาตำแหน่งไฟล์แบบ Dynamic
    if file_path.exists():  # ตรวจสอบว่าไฟล์มีอยู่จริง
        with open(file_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.error(f"CSS file not found: {file_path}")

# โหลด CSS
local_css("styles.css")  # ไม่ต้องใช้ "./" หรือ "../"
# เลือกหน้า (Home, ML Model, Neural Network Model, Demo Prediction)
page = st.sidebar.radio("เลือกหน้า", ("Machine Learning Model", "Neural Network Model", "Demo Machine Learning Model", "Demo Neural Network Model"))

# --- หน้าแรก (Home) ---
if page == "Machine Learning Model":
    display_home()

# --- หน้า Demo Prediction ---
elif page == "Neural Network Model":
    display_demo_prediction()

# --- หน้า ML Model ---
elif page == "Demo Machine Learning Model":
    display_ml_models()

# --- หน้า Neural Network Model ---
elif page == "Demo Neural Network Model":
    display_nn_model()

st.sidebar.markdown("---")
st.sidebar.markdown("**นางสาวอรวรา สวัสดิ์อุบล**")
st.sidebar.markdown("6604062610594")
st.sidebar.markdown("**นางสาวณิชารีย์ อยู่ศิริ**")
st.sidebar.markdown("6604062620255")
