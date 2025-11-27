
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import torch
from PIL import Image
from model_loader import load_model
from db import db, fs, save_image
from retrain_model import retrain_and_save
import pandas as pd
import random

# ─────────────────────── MODEL LOADING: ───────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner="Loading AI model... (only once)")
def get_model():
    return load_model() 
model, device, preprocess, version = get_model()

# ─────────────────────── PAGE SETUP ───────────────────────
st.set_page_config(page_title="Breast Cancer AI", layout="wide")
st.title("Breast cancer mammogram classifier")
st.sidebar.success("Dashboard")  

tab1, tab2, tab3 = st.tabs(["Prediction", "Dataset", "Retrain Model"])

# ─────────────────────── TAB 1: PREDICTION ───────────────────────
with tab1:
    st.header("Upload mammogram")
    uploaded = st.file_uploader("Choose image", type=["png", "jpg", "jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, use_column_width=True)
        with col2:
            with st.spinner("Analyzing..."):
                tensor = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    prob = torch.softmax(model(tensor), 1)[0]
                    conf = prob.max().item() * 100
                    pred = "Malignant" if prob[1] > prob[0] else "Benign"
                if pred == "Malignant":
                    st.error("Malignant — Cancer Detected")
                    st.warning("Consult a doctor immediately")
                else:
                    st.success("Benign — No Cancer")
                    st.balloons()
                st.progress(conf / 100)
                st.write(f"**Confidence: {conf:.1f}%**")

# ─────────────────────── TAB 2: DATASET ───────────────────────
with tab2:
    st.header("Dataset Overview")
    data_stats = {"Class": ["Benign", "Malignant", "Total"], "Count": [793, 2590, 3383], "Percentage": ["23.4%", "76.6%", "100%"]}
    st.table(pd.DataFrame(data_stats))
    st.info("Original test accuracy: **72.05%**")

    st.markdown("### Sample Images")
    benign_path = "data/test/0"
    malignant_path = "data/test/1"
    samples = []
    for path, label in [(benign_path, "Benign"), (malignant_path, "Malignant")]:
        if os.path.exists(path):
            files = random.sample(os.listdir(path), min(6, len(os.listdir(path))))
            for f in files:
                img = Image.open(os.path.join(path, f)).convert("RGB")
                samples.append((img, label))
    col_b, col_m = st.columns(2)
    with col_b:
        st.subheader("Benign Samples")
        for img, lbl in [s for s in samples if s[1] == "Benign"][:6]:
            st.image(img, width=180)
    with col_m:
        st.subheader("Malignant Samples")
        for img, lbl in [s for s in samples if s[1] == "Malignant"][:6]:
            st.image(img, width=180)

# ─────────────────────── TAB 3: RETRAIN  ───────────────────────
with tab3:
    st.header("Upload new images and retrain model")


    if 'upload_success_message' not in st.session_state:
        st.session_state.upload_success_message = None
    if 'upload_warning_message' not in st.session_state:
        st.session_state.upload_warning_message = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False

    MODEL_V2_PATH = "models/best_mammogram_v2.pth"
    if 'v2_saved_path' not in st.session_state:
        st.session_state.v2_saved_path = MODEL_V2_PATH

    try:
        new_count = db["mammograms.files"].count_documents({"split": "retrain"})
    except:
        new_count = 0
    st.metric("New breast cancer cases Added", new_count)

    with st.form("upload_form", clear_on_submit=True):
        uploaded_files = st.file_uploader("Upload new mammograms", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="retrain_upload")
        label_choice = st.radio("Label for all images", ["Benign (0)", "Malignant (1)"])
        submitted = st.form_submit_button("Save images to Database")
        
        if submitted:
            if not uploaded_files:
                st.session_state.upload_warning_message = "Please provide images to save ⚠️"
                st.session_state.upload_success_message = None
            else:
                label_int = 1 if "Malignant" in label_choice else 0
                prog = st.progress(0)
                
                for i, f in enumerate(uploaded_files):
                    try:
                        save_image(f, label_int)
                    except:
                        pass
                    prog.progress((i + 1) / len(uploaded_files))
                prog.empty()

                st.session_state.upload_success_message = f"Added {len(uploaded_files)} new case(s)! 💾"
                st.session_state.upload_warning_message = None
        
    if st.session_state.upload_warning_message:
        st.warning(st.session_state.upload_warning_message)
        st.session_state.upload_warning_message = None

    if st.session_state.upload_success_message:
        st.success(st.session_state.upload_success_message)
        st.session_state.upload_success_message = None 

    st.markdown("---")
    st.subheader("Retrain model")


    if st.button("Start retraining", type="primary", use_container_width=False, key="start_retrain_button"):
        if new_count < 5:
            st.warning("Add at least 5 new images")
        else:
            st.session_state.model_trained = False 
            st.info(f"Training on {new_count} new cases")

            progress_bar = st.progress(0)
            status_text = st.empty()
            log_box = st.code("")

            def update_progress(stage, percent):
                progress_bar.progress(percent / 100)
                status_text.markdown(f"**{stage}**")
                log_box.code(stage)

            error = retrain_and_save(
                db=db,
                fs=fs,
                model_path=MODEL_V2_PATH, 
                device=DEVICE,
                progress_callback=update_progress
            )

   
            if error:
                st.error(f"Training failed: {error}")
            else:
                st.success("Model successfully updated and saved as V2!")
                st.session_state.model_trained = True
          

    st.markdown("---")
    
    if st.session_state.model_trained:
        st.subheader("Re-training Complete 🎉")
        st.info("Model V2 was saved successfully! You can download it below.")
        
        if os.path.exists(MODEL_V2_PATH):
            with open(MODEL_V2_PATH, "rb") as file:
                st.download_button(
                    label="Download New Model (best_mammogram_v2.pth)",
                    data=file,
                    file_name="best_mammogram_v2.pth",
                    mime="application/octet-stream",
                    use_container_width=True
                )
        else:
             st.warning(f"Model file not found at {MODEL_V2_PATH}. Cannot provide download link.")
            
    else:
        st.info("Ready for continual learning")