

from db import db
from datetime import datetime

IMAGE_COLLECTION = "mammograms.files"

print(f"Connecting to Database: **{db.name}**")
print(f"LATEST 10 IMAGES IN COLLECTION: {IMAGE_COLLECTION} (by upload date)")
print("=" * 80)

cursor = db[IMAGE_COLLECTION].find() \
                           .sort("uploadDate", -1) \
                           .limit(10)

count = 0
for doc in cursor:
    count += 1
    filename = doc.get("filename", "Unknown")
    label = doc.get("label")
    label_text = "Malignant" if label == 1 else "Benign" if label == 0 else "Unknown"
    split = doc.get("split", "unknown")
    source = doc.get("source", "unknown")
    size_kb = doc.get("length", 0) // 1024
    upload_time = doc.get("uploadDate")

    if isinstance(upload_time, datetime):
        time_str = upload_time.strftime("%Y-%m-%d %H:%M:%S")
    else:
        time_str = str(upload_time)

    print(f"{count:2}. {filename}")
    print(f"    → Label: {label_text} | Split: {split} | Source: {source}")
    print(f"    → Size: {size_kb} KB")
    print(f"    → Uploaded: {time_str}")
    print("    → File ID:", doc.get("_id"))
    print("-" * 80)

total = db[IMAGE_COLLECTION].count_documents({})
print(f"\nTOTAL IMAGES IN DATABASE: {total}")

if total == 0:
    print(f"No images found in {db.name}.{IMAGE_COLLECTION}.")
else:
    print("DATABASE IS ALIVE AND WORKING!")