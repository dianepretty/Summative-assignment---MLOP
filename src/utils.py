# src/utils.py
from PIL import Image
import torch

def predict_image(model, device, preprocess, image):
    img = Image.open(image).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred = probs.argmax().item()
        confidence = probs[pred].item() * 100
    
    label = "Malignant (Cancer)" if pred == 1 else "Benign (No Cancer)"
    return label, confidence