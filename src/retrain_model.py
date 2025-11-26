import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import io
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import re
import os

# --- 1. DATA UTILITY FUNCTIONS ---

def get_all_data(db, fs):
    """Fetches all image data from GridFS and corrects/assigns labels.
    :param db: The MongoDB database object.
    :param fs: The GridFS object.
    """
    all_files = list(fs.find())
    correct_labels = []
    
    # Correcting labels using the filename rule (or metadata for new uploads)
    for f in all_files:
        # Attempt to infer label from old filename convention
        match = re.search(r'^(\d+)', f.filename)
        if match:
            first_num = int(match.group(1))
            label = 1 if first_num >= 1000 else 0 
        else:
            # For new uploads, use the 'label' metadata if available
            label = f.get('label', 0) 
        
        # Store the corrected label temporarily for the training run
        f.temp_corrected_label = label
        correct_labels.append(label)
        
    return all_files, correct_labels

class MammogramDataset(Dataset):
    """Custom Dataset for loading data from in-memory file objects via GridFS.
    It relies on the GridFS object (fs_instance) being passed during initialization.
    """
    def __init__(self, file_objects, fs_instance, transform=None): 
        self.file_objects = file_objects
        self.fs = fs_instance # Store the GridFS instance
        self.transform = transform

    def __len__(self):
        return len(self.file_objects)

    def __getitem__(self, idx):
        grid_file = self.file_objects[idx]
        
        # Re-fetch raw bytes from GridFS using the stored fs instance
        image_bytes = self.fs.get(grid_file._id).read() 
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        label = int(grid_file.temp_corrected_label)

        if self.transform:
            image = self.transform(image)
        return image, label

def cache_to_tensor(dataset):
    """Loads all data into PyTorch Tensors in RAM for fast training."""
    images, labels = [], []
    # Use DataLoader to fetch and process images from GridFS
    for img, lbl in DataLoader(dataset, batch_size=32, num_workers=0):
        images.append(img)
        labels.append(lbl)
    
    if not images:
        return torch.tensor([]), torch.tensor([])

    return torch.cat(images), torch.cat(labels)


# --- 2. CORE RETRAINING FUNCTION ---

# This function is what app.py imports
def retrain_and_save(db, fs, model_path, device, model_version_tag, progress_callback=None):
    """
    Executes the full training pipeline using all data in MongoDB.
    :param db: The MongoDB database object.
    :param fs: The GridFS object.
    :param model_path: The path to save the new model checkpoint.
    :param device: The PyTorch device to use ('cuda' or 'cpu').
    :param model_version_tag: A string to identify the new model version (e.g., 'v2').
    :param progress_callback: A function (stage_name, percent) to update Streamlit UI.
    :return: The test accuracy of the new model and an error message (or None).
    """
    
    def update_progress(stage, percent):
        """Updates the Streamlit progress bar via callback."""
        if progress_callback:
            # Ensure percentage stays within [0, 100]
            progress_callback(stage, min(100, max(0, int(percent))))

    # A. Data Loading and Preprocessing
    update_progress("Loading Data", 5)
    
    # 1. Fetch data and correct labels
    all_files, correct_labels = get_all_data(db, fs)
    total_images = len(all_files)
    if total_images < 10: # Minimum images needed for a split (approx)
        return 0.0, f"Not enough images found in the database. Found: {total_images}"

    # 2. Stratified Split (70/15/15)
    update_progress("Splitting Data", 15)
    indices = np.arange(total_images)
    
    # Attempt stratified split
    try:
        train_idx, temp_idx = train_test_split(indices, test_size=0.3, stratify=correct_labels, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=[correct_labels[i] for i in temp_idx], random_state=42)
    except ValueError:
        # Fallback for datasets too small or heavily skewed
        train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)


    # 3. Define Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 4. Create Datasets and Cache to RAM
    update_progress("Caching Data", 30)
    
    # Instantiate Datasets, passing the 'fs' object to MammogramDataset
    train_dataset = MammogramDataset([all_files[i] for i in train_idx], fs_instance=fs, transform=train_transform)
    val_dataset   = MammogramDataset([all_files[i] for i in val_idx],   fs_instance=fs, transform=val_test_transform)
    test_dataset  = MammogramDataset([all_files[i] for i in test_idx],  fs_instance=fs, transform=val_test_transform)
    
    train_X, train_y = cache_to_tensor(train_dataset)
    val_X,   val_y  = cache_to_tensor(val_dataset)
    test_X,  test_y = cache_to_tensor(test_dataset)
    
    if train_X.size(0) == 0:
        return 0.0, "Not enough data for training (0 training images after split)."

    # 5. Final ultra-fast in-memory loaders
    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=32, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(TensorDataset(val_X, val_y), batch_size=32, shuffle=False, pin_memory=True)
    test_loader  = DataLoader(TensorDataset(test_X, test_y), batch_size=32, shuffle=False, pin_memory=True)

    # B. Model Setup and Training
    update_progress("Initializing Model", 40)
    
    # EfficientNet-B0 with Transfer Learning
    from torchvision.models import efficientnet_b0
    model = efficientnet_b0(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(model.classifier[1].in_features, 2)
    )
    model = model.to(device)

    # Class Weights for balanced loss
    classes = np.unique(train_y.numpy())
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_y.numpy())
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    max_epochs = 10 
    patience = 3    
    best_acc = 0.0
    patience_counter = 0

    
    for epoch in range(max_epochs):
        
        # Train
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Progress update logic: 40% to 80% is training
            epoch_percent = (i + 1) / len(train_loader)
            total_epochs_progress = (epoch + epoch_percent) / max_epochs
            # 40% (start) + (40% total training block * progress)
            overall_percent = 40 + (total_epochs_progress * 40)
            update_progress(f"Training Epoch {epoch+1}/{max_epochs}", overall_percent)


        # Validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            # Ensure the directory exists before saving (e.g., 'models/')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            # Save the new best model
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            break

    # C. Final Testing
    update_progress("Final Testing", 90)
    
    # Load the best model saved during training
    try:
        # Load the best checkpoint from the designated path
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        return 0.0, f"Training complete, but model load failed for testing: {e}"

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    test_acc = 100 * sum(np.array(all_preds) == all_labels) / len(all_labels)
    update_progress("Finished", 100)

    return test_acc, None