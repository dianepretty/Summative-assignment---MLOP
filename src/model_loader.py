import streamlit as st
import torch
from torchvision import models, transforms
import torch.nn as nn
import os


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    model_path = os.path.join(base_dir, "models", "best_mammogram_model.pth")
    version = "V1 (Original - best_mammogram_model.pth)"
    
    if not os.path.exists(model_path):
        st.error(f"Required Model 1 file not found: {model_path}")
        st.info("Please ensure 'best_mammogram_model.pth' is in the **models/** folder.")
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