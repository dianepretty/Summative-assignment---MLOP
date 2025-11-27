
import pymongo
from gridfs import GridFS
from datetime import datetime, timezone
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("MONGO_URI not found in .env file!")

client = pymongo.MongoClient(MONGO_URI)
db = client["breast_cancer_db"]
fs = GridFS(db, collection="mammograms")

def save_image(file, label):
    """Save uploaded image to GridFS"""
    fs.put(
        file.read(),
        filename=file.name,
        label=int(label),
        split="retrain",
        uploadDate=datetime.now(timezone.utc)
    )