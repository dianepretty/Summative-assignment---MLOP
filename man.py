import sys
import os
# Added import for ObjectId for explicit type handling/debugging
try:
    from bson.objectid import ObjectId
except ImportError:
    # This try/except handles environments where 'bson' might not be directly available, 
    # though pymongo typically installs it.
    pass 

# Import the database connection objects (db and fs) which are assumed to be
# initialized and available in your db.py file.
try:
    from db import db, fs
except ImportError:
    print("Error: Could not find db.py. Please ensure the path is correct.")
    sys.exit(1)


def delete_retraining_cases(db, fs):
    """
    Finds and deletes ALL files marked with "split": "retrain".
    This resets the new data pool for the next retraining cycle.
    """
    
    print("Attempting to find and delete ALL retraining cases...")
    
    # Query for all files marked for retraining
    files_cursor = db["mammograms.files"].find({
        "split": "retrain"
    })

    # Convert cursor to a list of IDs to determine the total count upfront
    file_ids = [doc["_id"] for doc in files_cursor]
        
    if not file_ids:
        print("Success: 0 retraining cases found for deletion.")
        return 0
        
    deleted_files = 0
    total_to_delete = len(file_ids)
    
    print(f"Found {total_to_delete} retraining cases to delete. Starting deletion process...")

    for i, file_id in enumerate(file_ids):
        id_type = type(file_id).__name__
        print(f"Processing case {i + 1}/{total_to_delete}. ID Type: {id_type}, ID: {file_id}")
        
        try:
            # GridFS fs.delete() handles the deletion of both the file chunks
            # and the file metadata document from the 'mammograms.files' collection.
            fs.delete(file_id)
            
            deleted_files += 1
            print(f"Deleted case {i + 1}/{total_to_delete} successfully.")
            
        except Exception as e:
            # We catch exceptions here (e.g., if the ID is invalid)
            print(f"Error deleting file {file_id}: {e}")
            
    print(f"\n--- Deletion Complete ---")
    print(f"Total files processed: {total_to_delete}")
    print(f"Successfully deleted files: {deleted_files}")
    print("-------------------------")
    
    return deleted_files


def delete_old_malignant_cases(db, fs, count=50):
    """
    Deletes a specified count of older malignant cases (label=1) 
    that are NOT currently marked for retraining (split != 'retrain').
    
    NOTE: This function is kept for completeness but is not run by default.
    """
    
    print(f"Attempting to find and delete up to {count} old malignant cases...")
    
    # Query for malignant files that are NOT part of the new retraining set
    files_cursor = db["mammograms.files"].find({
        "label": 1,
        "split": {"$ne": "retrain"}
    }).limit(count)

    file_ids = [doc["_id"] for doc in files_cursor]
        
    if not file_ids:
        print("Success: 0 old malignant cases found for deletion.")
        return 0
        
    deleted_files = 0
    total_to_delete = len(file_ids)
    
    for i, file_id in enumerate(file_ids):
        try:
            # Use fs.delete() for comprehensive cleanup
            fs.delete(file_id)
            deleted_files += 1
        except Exception as e:
            print(f"Error deleting file {file_id}: {e}")
            
    return deleted_files


if __name__ == "__main__":
    
    print("--- Database Maintenance Script Started ---")
    
    try:
        # EXECUTE: Delete ALL retraining files as requested by the user
        delete_count = delete_retraining_cases(db, fs)
        
        if delete_count > 0:
            print(f"Maintenance finished. {delete_count} retraining files removed.")
        
    except Exception as e:
        print(f"A critical error occurred during script execution: {e}")
        sys.exit(1)