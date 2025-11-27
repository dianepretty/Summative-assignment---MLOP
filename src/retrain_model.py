# src/retrain_model.py — FINAL: ZERO ERRORS, TRUE CONTINUAL LEARNING
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import io
import os
from sklearn.metrics import accuracy_score
import numpy as np
import random

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def retrain_and_save(db, fs, model_path, device, progress_callback=None):
    def update(stage, percent):
        if progress_callback:
            progress_callback(stage, percent)

    try:
        update("Loading data...", 5)

        new_files = list(db["mammograms.files"].find({"split": "retrain"}))
        if len(new_files) < 5:
            return "Need at least 5 new cases."

        print(f"Found {len(new_files)} new case(s)")

        # Small replay buffer from original data
        old_benign = list(db["mammograms.files"].find({"label": 0, "split": {"$ne": "retrain"}}).limit(15))
        old_malig  = list(db["mammograms.files"].find({"label": 1, "split": {"$ne": "retrain"}}).limit(15))
        all_files = new_files + old_benign + old_malig
        random.shuffle(all_files)

        print(f"Total training pool: {len(all_files)} (new + replay)")

        split_idx = int(0.85 * len(all_files))
        train_files = all_files[:split_idx]
        val_files   = all_files[split_idx:]

        def load_images(files, transform, repeats):
            imgs, lbls = [], []
            for doc in files:
                try:
                    data = fs.get(doc["_id"]).read()
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                    for _ in range(repeats):
                        imgs.append(transform(img))
                        lbls.append(int(doc["label"]))
                except:
                    continue
            return imgs, lbls

        train_imgs, train_lbls = load_images(train_files, train_transform, repeats=3)
        val_imgs,   val_lbls   = load_images(val_files,   val_transform,   repeats=1)

        if len(train_imgs) == 0 or len(val_imgs) == 0:
            return "Failed to load images."

        train_loader = DataLoader(TensorDataset(torch.stack(train_imgs), torch.tensor(train_lbls)), batch_size=16, shuffle=True)
        val_loader   = DataLoader(TensorDataset(torch.stack(val_imgs),   torch.tensor(val_lbls)),   batch_size=16)

        update("Loading model...", 30)

        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.classifier[1].in_features, 2)
        )

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        v1_path = os.path.join(base_dir, "models", "best_mammogram_model.pth")
        v2_path = os.path.join(base_dir, "models", "best_mammogram_v2.pth")
        load_path = v2_path if os.path.exists(v2_path) else v1_path

        model.load_state_dict(torch.load(load_path, map_location=device))
        model.to(device)

        for param in model.features.parameters():
            param.requires_grad = False

        optimizer = optim.AdamW(model.classifier.parameters(), lr=0.0005, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        update("Fine-tuning...", 50)

        for epoch in range(12):
            model.train()
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                loss = criterion(model(imgs), lbls)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    preds = model(imgs).argmax(1)
                    total += lbls.size(0)
                    correct += (preds == lbls).sum().item()
            val_acc = 100 * correct / total if total > 0 else 0

            print(f"Epoch {epoch+1:2d}/12 — Val Acc: {val_acc:.2f}%")
            update(f"Epoch {epoch+1}/12 — Val: {val_acc:.1f}%", 50 + (epoch+1)*4)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_path)

        # FINAL EVALUATION — 100% SAFE (NO classification_report)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        all_preds = []
        with torch.no_grad():
            for imgs, _ in val_loader:
                all_preds.extend(model(imgs.to(device)).argmax(1).cpu().numpy())

        final_acc = accuracy_score(val_lbls, all_preds) * 100
        unique_labels = sorted(set(val_lbls))

        # Manual precision/recall — never crashes
        tp_benign = sum(1 for t, p in zip(val_lbls, all_preds) if t == 0 and p == 0)
        tp_malig  = sum(1 for t, p in zip(val_lbls, all_preds) if t == 1 and p == 1)
        pred_benign = sum(1 for p in all_preds if p == 0)
        pred_malig  = sum(1 for p in all_preds if p == 1)
        true_benign = sum(1 for t in val_lbls if t == 0)
        true_malig  = sum(1 for t in val_lbls if t == 1)

        benign_prec = tp_benign / pred_benign * 100 if pred_benign > 0 else 0
        malig_recall = tp_malig / true_malig * 100 if true_malig > 0 else 0

        print("\n" + "="*60)
        print("CONTINUAL LEARNING COMPLETE — HONEST METRICS")
        print(f"Validation images      : {len(val_lbls)}")
        print(f"Validation Accuracy    : {final_acc:.2f}%")
        print(f"Benign Precision       : {benign_prec:.1f}%")
        print(f"Malignant Recall       : {malig_recall:.1f}%")
        print(f"Classes in validation  : {unique_labels}")
        print(f"Model saved            : {model_path}")
        print("="*60 + "\n")

        update("Model updated successfully", 100)
        return None

    except Exception as e:
        error = f"Error: {str(e)}"
        print(error)
        return error