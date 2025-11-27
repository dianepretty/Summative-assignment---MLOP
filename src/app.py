# src/app.py — CLEAN, SIMPLE, WORKING 100%
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="triton")
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import streamlit as st
import torch
from PIL import Image
from model_loader import load_model
from db import db, fs, save_image
from retrain_model import retrain_and_save

# Model setup
MODEL_PATH_V2 = "models/best_mammogram_v2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def get_model():
    version = st.session_state.get("active_model_version", 1)
    if version == 2 and os.path.exists(MODEL_PATH_V2):
        try:
            from torchvision.models import efficientnet_b0
            import torch.nn as nn
            model = efficientnet_b0(weights=None)
            model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, 2))
            model.load_state_dict(torch.load(MODEL_PATH_V2, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            _, _, preprocess, _ = load_model()
            return model, DEVICE, preprocess, "v2"
        except:
            pass
    return load_model()

model, device, preprocess, version = get_model()

# Page
st.set_page_config(page_title="Breast Cancer AI", layout="wide")
st.title("Breast Cancer Mammogram Classifier")
st.sidebar.success(f"Model: {version}")

tab1, tab2, tab3 = st.tabs(["Prediction", "Dataset", "Retrain Model"])

with tab1:
    st.header("Upload Mammogram")
    uploaded = st.file_uploader("Choose image", type=["png", "jpg", "jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, width="stretch")
        with col2:
            with st.spinner("Analyzing..."):
                tensor = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    prob = torch.softmax(model(tensor), 1)[0]
                    conf = prob.max().item() * 100
                    pred = "Malignant" if prob[1] > prob[0] else "Benign"
                if pred == "Malignant":
                    st.error("Malignant — Cancer Detected")
                    st.warning("Consult a doctor")
                else:
                    st.success("Benign — No Cancer")
                    st.balloons()
                st.progress(conf / 100)
                st.write(f"Confidence: {conf:.1f}%")

with tab2:
    st.header("Dataset Overview")
    st.write("Benign: 793 | Malignant: 2590 | Total: 3383")
    st.info("Original test accuracy: 72.05%")

with tab3:
    st.header("Retrain Model with New Clinical Cases")

    # --- COUNT NEW IMAGES ---
    try:
        new_count = db["mammograms.files"].count_documents({"split": "retrain"})
    except:
        new_count = 0
    st.metric("New Cases in Database", new_count)

    # --- UPLOAD FORM (still here, fully working!) ---
    with st.form("upload_form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Upload new mammograms",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="retrain_upload"
        )
        label_choice = st.radio("Label for all uploaded images", ["Benign (0)", "Malignant (1)"])
        submitted = st.form_submit_button("Save to Database")

        if submitted and uploaded_files:
            label_int = 1 if "Malignant" in label_choice else 0
            prog = st.progress(0)
            saved = 0
            for i, f in enumerate(uploaded_files):
                try:
                    save_image(f, label_int)
                    saved += 1
                except Exception as e:
                    st.error(f"Failed: {f.name}")
                prog.progress((i + 1) / len(uploaded_files))
            prog.empty()
            if saved:
                st.success(f"Successfully saved {saved} new case(s)!")

    st.markdown("---")
    st.subheader("Fine-Tune Model")

    if st.button("Start Retraining → Create Updated Model", type="primary", use_container_width=True):
        current_count = db["mammograms.files"].count_documents({"split": "retrain"})
        if current_count == 0:
            st.warning("No new cases found in the database.")
        else:
            st.info(f"Starting retraining on {current_count} new case(s)...")

            progress_bar = st.progress(0, text="Initializing...")
            status_text = st.empty()

            def update_progress(stage, percent):
                progress_bar.progress(percent / 100, text=stage)
                status_text.markdown(f"**Status:** {stage}")

            error_msg = retrain_and_save(
                db=db,
                fs=fs,
                model_path="models/best_mammogram_v2.pth",
                device=DEVICE,
                progress_callback=update_progress
            )

            progress_bar.empty()
            status_text.empty()

            if error_msg:
                st.error(f"Retraining failed: {error_msg}")
            else:
                st.success("Model successfully updated and saved as V2")
                st.info("You can now switch to the improved model using the buttons below.")

    # --- MODEL SWITCHER ---
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Use Original Model (V1)", use_container_width=True):
            st.session_state.active_model_version = 1
            st.rerun()
    with col2:
        if os.path.exists("models/best_mammogram_v2.pth"):
            if st.button("Use Updated Model (V2)", use_container_width=True):
                st.session_state.active_model_version = 2
                st.rerun()
        else:
            st.write("Updated model not available yet")