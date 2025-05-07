import os
import cv2
import numpy as np

def categorize_age(age):
    """Categorize age into groups"""
    if age < 13:
        return "child"
    elif age < 20:
        return "teen"
    elif age < 60:
        return "adult"
    else:
        return "senior"

def extract_age_from_filename(filename):
    """Extract actual age from filename if available (for calibration)"""
    # Assumes filename format like: 35_1_0_20170109205304456.jpg
    # Where 35 is the actual age
    try:
        parts = filename.split('_')
        if len(parts) >= 1 and parts[0].isdigit():
            return int(parts[0])
    except:
        pass
    return None 