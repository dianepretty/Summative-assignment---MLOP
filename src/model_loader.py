# src/model_loader.py — FINAL WORKING VERSION
import streamlit as st
import torch
from torchvision import models, transforms
import torch.nn as nn
import os

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GO UP ONE LEVEL FROM src/ TO PROJECT ROOT
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    v2_path = os.path.join(base_dir, "models", "best_mammogram_v2.pth")
    v1_path = os.path.join(base_dir, "models", "best_mammogram_model.pth")

    if os.path.exists(v2_path):
        model_path = v2_path
        version = "v2 (Improved)"
    elif os.path.exists(v1_path):
        model_path = v1_path
        version = "v1 (Original)"
    else:
        st.error("Model file not found!")
        st.info("Put your model in the **models/** folder (same level as src/)")
        st.stop()

    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.classifier[1].in_features, 2)
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, device, preprocess, version