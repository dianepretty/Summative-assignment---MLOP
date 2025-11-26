# src/app.py — FINAL 100% WORKING (UPLOAD & RETRAIN ADDED)
import streamlit as st
import torch
from PIL import Image
import os
from model_loader import load_model
from db import db, fs, save_image
from datetime import datetime, timezone
from retrain_model import retrain_and_save 

# Define model path and device
MODEL_DIR = "models"
MODEL_PATH_V2 = f"{MODEL_DIR}/model_v2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load the current active model version
@st.cache_resource
def get_current_model(model_version=1):
    """Loads the model, preferring V2 if it exists and is requested."""
    # Check if a new version (V2) exists
    if model_version == 2 and os.path.exists(MODEL_PATH_V2):
        # Manually reconstruct the EfficientNet-B0 to load the state dict
        try:
            from torchvision.models import efficientnet_b0
            import torch.nn as nn
            
            # Initialize the base model architecture
            model = efficientnet_b0(pretrained=False) 
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(model.classifier[1].in_features, 2)
            )
            # Load the V2 weights
            model.load_state_dict(torch.load(MODEL_PATH_V2, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            
            # Assuming 'preprocess' from V1 is cached or accessible
            preprocess = st.session_state.get('preprocess_v1')
            if not preprocess:
                # If cached preprocess is missing, fall back to initial load to get it
                _, _, preprocess, _ = load_model()

            return model, DEVICE, preprocess, "v2"
        except Exception as e:
            st.warning(f"Error loading V2 model: {e}. Falling back to V1.")
            
    # Fallback/Initial load (V1)
    model, device, preprocess, version = load_model()
    # Cache preprocess if it's the first time
    if 'preprocess_v1' not in st.session_state:
        st.session_state.preprocess_v1 = preprocess
    return model, device, preprocess, version


# Load model (initial or V2)
# Determine which model version to load based on session state
current_version = st.session_state.get('active_model_version', 1)
model, device, preprocess, version = get_current_model(current_version)


# Page setup
st.set_page_config(page_title="Breast Cancer AI", page_icon="doctor", layout="wide")
st.title("Breast Cancer Mammogram Classifier")

# Sidebar
st.sidebar.title("Dashboard")
st.sidebar.success(f"Model: {version}")
st.sidebar.caption("EfficientNet-B0 • 72.05% accuracy (V1)")
if version == "v2" and 'v2_test_accuracy' in st.session_state:
    st.sidebar.caption(f"V2 Test Accuracy: {st.session_state.v2_test_accuracy:.3f}%")

# Tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "Dataset Overview", "Retrain Model"])

with tab1:
    st.header("Upload Mammogram")
    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"], key="pred")

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded", use_column_width=True)
        with col2:
            st.write("### AI Diagnosis")
            with st.spinner("Analyzing..."):
                tensor = preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(tensor)
                    probs = torch.softmax(output, dim=1)[0]
                    confidence = probs.max().item() * 100
                    pred = "Malignant" if probs[1] > probs[0] else "Benign"

                if pred == "Malignant":
                    st.error("**Malignant — Cancer Detected**")
                    st.warning("Please consult a doctor")
                else:
                    st.success("**Benign — No Cancer**")
                    # st.balloons() # Removed per user request
                st.progress(confidence / 100)
                st.write(f"**Confidence:** {confidence:.2f}%")

with tab2:
    st.header("Dataset Overview")
    st.markdown("| Class | Meaning | Count |\n|-------|---------|-------|\n| Benign | No cancer | 793 |\n| Malignant | Cancer | 2590 |")
    st.info("Total: 3383 images • Test Accuracy: 72.05%")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    benign = os.path.join(base_dir, "data", "test", "0", "5_940388076_png.rf.b61a6986ef24d48d9dabfdc4a4947ebb.jpg")
    malignant = os.path.join(base_dir, "data", "test", "1", "729_748167281_png.rf.496a8581ecdbc0dfd0ffbb1affc2a1d1.jpg")

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(benign):
            st.image(benign, caption="Benign", use_column_width=True)
            st.success("Safe")
        else:
            st.write("Benign example")
    with col2:
        if os.path.exists(malignant):
            st.image(malignant, caption="Malignant", use_column_width=True)
            st.error("Cancer")
        else:
            st.write("Malignant example")

# ———————————————————— RETRAIN TAB — INCLUDING VISUAL PROGRESS ————————————————————
with tab3:
    st.header("Retrain Model with New Data")
    st.success("Upload real cases → make the AI smarter over time")

    # Placeholder for the Metric
    metric_placeholder = st.empty()

    # LIVE COUNT - Initial Fetch and Session State Initialization
    if 'new_upload_count' not in st.session_state:
        try:
            # We assume 'db' and 'fs' are available through the Canvas environment (via db.py)
            st.session_state.new_upload_count = fs.files.count_documents({}) # Count all files in GridFS
        except:
            st.session_state.new_upload_count = 0
            
    # Display the metric using the value from Session State
    metric_placeholder.metric("Total Images for Training", st.session_state.new_upload_count)

    # UPLOAD FORM
    with st.form("retrain_upload", clear_on_submit=False):
        uploaded_files = st.file_uploader(
            "Upload new mammograms",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="retrain_files"
        )
        label_choice = st.radio("Label for all images", ["Benign (0)", "Malignant (1)"])
        submit = st.form_submit_button("SAVE TO DATABASE")

        # Message placeholder to show status without a rerun
        message_placeholder = st.empty()

        if submit and uploaded_files:
            message_placeholder.empty()

            label_int = 1 if "Malignant" in label_choice else 0
            progress = st.progress(0)
            saved_count = 0

            for i, uploaded_file in enumerate(uploaded_files):
                uploaded_file.seek(0)
                try:
                    save_image(uploaded_file, label_int)
                    saved_count += 1
                except Exception as e:
                    message_placeholder.error(f"Failed: {uploaded_file.name} → {e}")
                
                progress.progress((i + 1) / len(uploaded_files))
            
            progress.empty()

            if saved_count > 0:
                message_placeholder.success(f"SUCCESS: {saved_count} images saved to MongoDB!")
                
                # Update total image count
                st.session_state.new_upload_count += saved_count
                metric_placeholder.metric("Total Images for Training", st.session_state.new_upload_count)
            else:
                message_placeholder.error("No images were saved.")

    st.markdown("---")
    st.subheader("Model Retraining")
    
    # Get total count dynamically
    total_db_count = st.session_state.new_upload_count
        
    st.info(f"Retraining will use all data in the database (total images available: {total_db_count}) to create **Version 2**. The entire dataset must be loaded for this process.")
    
    # Retrain button and logic
    is_training = st.session_state.get('is_training', False)
    retrain_button = st.button("🚀 START RETRAINING (V2)", key="start_retrain", disabled=is_training)
    retrain_status = st.empty() # Used for overall success/fail message

    if retrain_button:
        # Prevent multiple clicks while training is active
        st.session_state.is_training = True
        
        # --- Start the visual loader using st.spinner ---
        with st.spinner("Model Retraining in Progress... Do not interrupt or navigate away."):
            
            # This progress bar will be updated by the callback function
            train_progress_bar = st.progress(0, text="Initializing training environment...")
            
            def training_progress_callback(stage, percent):
                """Callback function to update the Streamlit UI."""
                # The key element showing progress dynamically
                train_progress_bar.progress(percent, text=f"{stage} ({int(percent)}%)")

            try:
                # Call the retrain function
                test_acc, error_message = retrain_and_save(
                    db=db,
                    fs=fs,
                    model_path=MODEL_PATH_V2, 
                    device=DEVICE, 
                    model_version_tag="v2", 
                    progress_callback=training_progress_callback
                )
                
                # Clear the progress bar after completion
                train_progress_bar.empty() 
                
                if error_message:
                    retrain_status.error(f"Retraining Failed: {error_message}")
                elif test_acc > 0:
                    # Visual Feedback: Success message
                    retrain_status.success(f"✅ Training Complete! New Model V2 Test Accuracy: {test_acc:.3f}%")
                    st.session_state.active_model_version = 2
                    st.session_state.v2_test_accuracy = test_acc
                    st.balloons()
                else:
                    retrain_status.warning("Training finished but resulted in 0% accuracy or failed silently.")

            except Exception as e:
                retrain_status.error(f"An unexpected error occurred during training: {e}")
            finally:
                # Ensure button can be clicked again and reload the page to apply new model
                st.session_state.is_training = False 
                st.rerun()