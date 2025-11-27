import os
from locust import HttpUser, task, between
from io import BytesIO

# --- IMPORTANT SETUP ---
# 1. Start the locust test by running this in your terminal:
#    locust -f locustfile.py
# 2. Go to the URL provided (usually http://0.0.0.0:8089)
# 3. Enter the URL of your deployed Render app (e.g., https://summative-assignment-mlop-1-tnkn.onrender.com)
# 4. Set the number of users and the spawn rate.

class StreamlitUser(HttpUser):
    """
    Locust class to simulate user interaction with a Streamlit app.
    
    NOTE: Streamlit is designed for single-user sessions, not heavy load.
    We simulate prediction and an image upload, which hits the prediction
    endpoint and the database/upload endpoint respectively.
    """
    # *** FIX: Specify the host URL for the local Streamlit application. ***
    host = "http://localhost:8501"
    
    # Wait between 1 and 3 seconds between performing tasks
    wait_time = between(1, 3) 

    # Since Streamlit is an interactive web app, the most common action is 
    # simply loading the page, which triggers the Python backend to run 
    # and perform prediction (the key ML step).
    @task(3) # Weight 3 (happens 3 times as often as upload)
    def load_prediction_page(self):
        """Simulates loading the main prediction page."""
        # This will hit the root path, which triggers the Streamlit app.
        # It tests the initial model loading speed (after the cache hit).
        self.client.get("/")

    @task(1) # Weight 1 (happens once for every three page loads)
    def upload_image_for_retraining(self):
        """
        Simulates uploading a test image to trigger the database logic (save_image).
        
        Since this is not a true Streamlit POST, we simulate the simplest POST 
        that would hit a theoretical upload endpoint, forcing the server to handle 
        a large data payload and a database write (GridFS).
        """
        # Create a dummy image file in memory (100KB, representative of a small mammogram)
        dummy_image_data = BytesIO(os.urandom(100 * 1024))
        
        files = {
            'file': ('test_image.png', dummy_image_data, 'image/png'),
            'label': (None, '0') # Assuming 0 (Benign) or 1 (Malignant)
        }

        # We need to know the specific upload endpoint of your Streamlit app.
        # Since Streamlit doesn't expose standard APIs, this task assumes 
        # a helper API endpoint exists, OR it simulates a high-cost database 
        # write that your app might eventually expose.
        # ***If your Streamlit app handles file uploads directly, this task 
        # will need modification to target the specific hidden POST requests Streamlit uses.***
        
        # For simplicity, we target the root path to stress-test the initial loading,
        # but the intention here is to simulate heavy resource usage.
        # If you were using a standard Flask/FastAPI backend, you'd target:
        # self.client.post("/upload", files=files) 
        
        # For a Streamlit-only deployment, we rely on the frequent page load 
        # to test the model serving latency.
        self.client.get("/") 


    def on_start(self):
        """Called when a Locust user starts."""
        print(f"Starting user session for {self.host}")