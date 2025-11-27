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

# Set torch threads once at startup
torch.set_num_threads(1)

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
    st.header(f"Upload mammogram ") 
    uploaded = st.file_uploader("Choose image", type=["png", "jpg", "jpeg"])
    
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img, width='stretch')
            
        with col2:
            with st.spinner("Analyzing..."):
                # Removed torch.set_num_threads(1) from here
                tensor = preprocess(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    prob = torch.softmax(model(tensor), 1)[0]
                    conf = prob.max().item() * 100
                    pred = "Malignant" if prob[1] > prob[0] else "Benign"
                    
                if pred == "Malignant":
                    st.error("Malignant — Cancer Detected 🚨")
                    st.warning("Consult a doctor immediately")
                else:
                    st.success("Benign — No Cancer ✅")
                    
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
            files = os.listdir(path)
            files = random.sample(files, min(6, len(files)))
            for f in files:
                try:
                    img = Image.open(os.path.join(path, f)).convert("RGB")
                    samples.append((img, label))
                except Exception:
                    continue 

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
    st.metric("New breast cancer cases added", new_count)

    # --- Upload Form ---
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
                    except Exception as e:
                        print(f"Failed to save file {f.name}: {e}")
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


    if st.button("Start retraining", type="primary", width='content', key="start_retrain_button"):
        if new_count < 5:
            st.warning("Add at least 5 new images before retraining.")
        else:
            st.session_state.model_trained = False 

            progress_bar = st.progress(0, text="Initializing...")
            status_text = st.empty()
            log_box = st.code("Starting process...")
            
            status_text.info(f"Training started on {new_count} new cases. This may take a few minutes...")

            def update_progress(stage, percent):
                """Callback function passed to the backend training script."""
                progress_bar.progress(percent / 100, text=stage)
                status_text.markdown(f"**Current Stage: {stage}**") 
                log_box.code(stage, language="log") 


            error = retrain_and_save(
                db=db,
                fs=fs,
                model_path=MODEL_V2_PATH, 
                device=DEVICE,
                progress_callback=update_progress
            )

            if error:
                status_text.error(f"Training failed: {error}")
                progress_bar.progress(1.0, text="Failed")
                log_box.code(f"Training failed: {error}", language="log")
            else:
                status_text.success("Model successfully updated and saved as V2! 🎉")
                progress_bar.progress(1.0, text="Complete!")
                log_box.code("Training complete. Model V2 saved.", language="log")
                st.session_state.model_trained = True
            
            
    st.markdown("---")
    
    if st.session_state.model_trained:
        st.subheader("Re-training Complete 🎉")
        st.info("Model V2 was saved successfully! You can download it below.")
        
        if os.path.exists(MODEL_V2_PATH):
            try:
                with open(MODEL_V2_PATH, "rb") as file:
                    st.download_button(
                        label="Download New Model (best_mammogram_v2.pth)",
                        data=file,
                        file_name="best_mammogram_v2.pth",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Could not open V2 file for download: {e}")
        else:
            st.warning(f"Model file not found at {MODEL_V2_PATH}. Cannot provide download link.")
            
    else:
        st.info("Ready for continual learning")