# src/model.py — FINAL WORKING VERSION
import torch
from torchvision import models
import torch.nn as nn
import os

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ABSOLUTE PATH — THIS IS THE MAGIC
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # goes up from src/ to project root
    v2_path = os.path.join(base_dir, "models", "best_mammogram_v2.pth")
    v1_path = os.path.join(base_dir, "models", "best_mammogram_model.pth")

    if os.path.exists(v2_path):
        model_path = v2_path
        print("Loading v2 (improved) model")
    elif os.path.exists(v1_path):
        model_path = v1_path
        print("Loading original v1 model")
    else:
        raise FileNotFoundError("No model found! Put best_mammogram_model.pth in the 'models' folder")

    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.classifier[1].in_features, 2)
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, device