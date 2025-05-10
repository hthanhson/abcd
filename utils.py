import os
import cv2
import numpy as np

def categorize_age(age):
    """
    Simplified age categorization function for backward compatibility.
    Since we're removing age groups from our algorithm, this just returns
    'adult' for all ages (15-53) to maintain compatibility with existing code.
    """
    # Since we're removing age groups, just return 'adult' for all ages
    # This maintains compatibility with existing code while removing age group functionality
    return "adult"

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