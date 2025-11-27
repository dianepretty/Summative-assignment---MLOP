# locustfile.py - Custom Locust Client for Direct Function Testing

import os
import random
import time
from locust import User, task, events, between

from src.model_loader import load_model 


# ─────────────────────── 1. THE CUSTOM CLIENT CLASS ───────────────────────

class PythonClient:
    """
    A custom client that runs Python functions directly instead of making HTTP calls.
    It simulates network latency/overhead using a sleep, if desired.
    """
    
    def __init__(self, environment):
        self.environment = environment

        try:
            # Load the model and preprocessing steps
            self.model, self.device, self.preprocess, self.version = load_model()
            self.model.eval()
            print("Model loaded successfully for Locust client.")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load model for Locust: {e}")
            raise e

    def execute_function(self, task_name, func, *args, **kwargs):
        """Helper to run a function and fire Locust events."""
        start_time = time.time()
        start_perf_counter = time.perf_counter()
        
        try:
       
            result = func(*args, **kwargs)

            response_time = (time.perf_counter() - start_perf_counter) * 1000
            

            events.request.fire(
                request_type="PYTHON",
                name=task_name,
                response_time=response_time,
                response_length=0 
            )
            return result
        except Exception as e:
            total_time = (time.perf_counter() - start_perf_counter) * 1000
            
         
            events.request.fire(
                request_type="PYTHON",
                name=task_name,
                response_time=total_time,
                exception=e
            )
            return None

# ─────────────────────── 2. THE CUSTOM USER CLASS ───────────────────────

class PythonUser(User):
    """
    A Locust user that uses the PythonClient to run functions.
    This simulates user actions against the core computational engine.
    """
    abstract = True

    wait_time = between(1, 3) 

    client = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.client is None:
            self.client = PythonClient(self.environment)
            

        self.BENIGN_IMAGE_PATH = os.path.join("data", "test", "0")
        self.MALIGNANT_IMAGE_PATH = os.path.join("data", "test", "1")
        
        self.sample_files = self._get_image_paths()

    def _get_image_paths(self):
        """Helper to collect a list of absolute image paths."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        benign_dir = os.path.join(base_dir, self.BENIGN_IMAGE_PATH)
        malignant_dir = os.path.join(base_dir, self.MALIGNANT_IMAGE_PATH)
        
        paths = []
        if os.path.exists(benign_dir):
            paths.extend([os.path.join(benign_dir, f) for f in os.listdir(benign_dir)])
        if os.path.exists(malignant_dir):
            paths.extend([os.path.join(malignant_dir, f) for f in os.listdir(malignant_dir)])
        
        if not paths:
            print("WARNING: No image files found for stress test!")
            
        return paths

    @task(3)
    def predict_mammogram_task(self):
        """Simulates a user uploading an image for real-time prediction (Tab 1)."""
        if not self.sample_files:
            return
            
        image_path = random.choice(self.sample_files)
        
        def predict_func():
    
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
            tensor = self.client.preprocess(img).unsqueeze(0).to(self.client.device)
            
            with torch.no_grad():
                prob = torch.softmax(self.client.model(tensor), 1)[0]
                pred = "Malignant" if prob[1] > prob[0] else "Benign"
            return pred

        self.client.execute_function(
            task_name="PREDICT / Inference",
            func=predict_func
        )

    @task(2)
    def save_new_case_task(self):
        """Simulates a user saving a new, labeled case to the database (Tab 3)."""

        def save_func():
       
            time.sleep(0.5) 
            return True

        self.client.execute_function(
            task_name="SAVE / DB Write",
            func=save_func
        )

    @task(1)
    def initiate_retraining_task(self):
        """Simulates an administrator initiating the model retraining (Tab 3)."""

        def retrain_func():
        
            time.sleep(2) 
            return True

        self.client.execute_function(
            task_name="RETRAIN / Job Trigger",
            func=retrain_func
        )


class MLComputationalUser(PythonUser):
    pass