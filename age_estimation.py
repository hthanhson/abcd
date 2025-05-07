import cv2
import numpy as np
import os
from utils import categorize_age, extract_age_from_filename

# Default age calibration parameters
age_calibration = {
    'multiplier': 1.0,  # Overall multiplier
    'offset': 0,        # Offset to add/subtract
    'child_factor': 1.0,    # Adjustment factor for children
    'teen_factor': 1.0,     # Adjustment factor for teenagers
    'adult_factor': 1.0,    # Adjustment factor for adults
    'senior_factor': 1.0,   # Adjustment factor for seniors
    'calibrated': False     # Flag indicating if calibration was done
}

def calibrate_age_estimation(data_folder, limit=100):
    """Calibrate age estimation based on files with known ages in their names"""
    global age_calibration
    
    image_files = [f for f in os.listdir(data_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Collect pairs of actual age and estimated age
    age_pairs = []
    age_errors_by_group = {
        'child': [],
        'teen': [],
        'adult': [],
        'senior': []
    }
    
    for img_file in image_files[:limit]:  # Limit to 100 images to optimize time
        # Try to extract actual age from filename
        actual_age = extract_age_from_filename(img_file)
        if actual_age is None:
            continue
        
        # Extract features and estimate age
        img_path = os.path.join(data_folder, img_file)
        try:
            import face_recognition
            from deepface import DeepFace
            
            image = face_recognition.load_image_file(img_path)
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                continue
                
            # Use DeepFace to estimate age
            analysis = DeepFace.analyze(img_path, actions=['age'], enforce_detection=False)
            raw_age = analysis[0]['age']
            
            # Extract face region
            top, right, bottom, left = face_locations[0]
            face_image = image[top:bottom, left:right]
            
            # Initial age estimation (without overall calibration)
            estimated_age = adjust_age_estimation(raw_age, face_image)
            
            # Save age pair for analysis
            age_pairs.append((actual_age, estimated_age))
            
            # Classify by age group
            age_group = categorize_age(actual_age)
            age_errors_by_group[age_group].append(actual_age - estimated_age)
        except Exception as e:
            print(f"Error calibrating with {img_file}: {e}")
    
    # If not enough data, don't calibrate
    if len(age_pairs) < 10:
        print("Not enough data for age calibration")
        return False
    
    # Calculate overall calibration factors using linear regression
    actuals = np.array([pair[0] for pair in age_pairs])
    estimates = np.array([pair[1] for pair in age_pairs])
    
    # Calculate best multiplier and offset
    if len(actuals) > 1:
        slope, intercept = np.polyfit(estimates, actuals, 1)
        age_calibration['multiplier'] = slope
        age_calibration['offset'] = intercept
    
    # Calculate adjustment factors for each age group
    for group, errors in age_errors_by_group.items():
        if len(errors) > 5:  # Need at least 5 samples for age group
            avg_error = np.mean(errors)
            if group == 'child':
                age_calibration['child_factor'] = 1.0 + avg_error / 10
            elif group == 'teen':
                age_calibration['teen_factor'] = 1.0 + avg_error / 20
            elif group == 'adult':
                age_calibration['adult_factor'] = 1.0 + avg_error / 40
            elif group == 'senior':
                age_calibration['senior_factor'] = 1.0 + avg_error / 50
    
    # Mark as calibrated
    age_calibration['calibrated'] = True
    print(f"Age calibration completed with {len(age_pairs)} samples")
    print(f"Calibration factors: {age_calibration}")
    
    return True

def adjust_age_estimation(age, face_image):
    """Improved age estimation using multiple facial features and advanced image analysis"""
    # Convert age to float for precise calculations
    age = float(age)
    
    # Check initial age from DeepFace
    if age <= 0:
        age = 25  # Default value if DeepFace returns invalid age (increased from 15 to 25)
    
    # Print info for debugging
    print(f"Initial DeepFace age: {age}")
    
    try:
        # Get face dimensions
        height, width = face_image.shape[:2]
        face_size_ratio = width / height
        
        # 1. Detailed facial proportions analysis
        # Convert to grayscale for feature detection
        gray_face = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding to enhance features
        thresh = cv2.adaptiveThreshold(gray_face, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
        
        # 2. Advanced wrinkle and texture analysis
        # Horizontal edge detection (for wrinkles)
        sobelx = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        sobel_mean = np.mean(abs_sobelx)
        
        # Vertical edge detection (for facial structure)
        sobely = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobely = np.absolute(sobely)
        sobely_mean = np.mean(abs_sobely)
        
        # Texture variance (skin smoothness)
        texture_variance = np.var(gray_face)
        
        # Calculate wrinkle density for forehead, eye area, and mouth area
        forehead_region = gray_face[0:int(height/3), :]
        eye_region = gray_face[int(height/3):int(2*height/3), :]
        mouth_region = gray_face[int(2*height/3):, :]
        
        forehead_edges = cv2.Sobel(forehead_region, cv2.CV_64F, 1, 0, ksize=3)
        eye_edges = cv2.Sobel(eye_region, cv2.CV_64F, 1, 0, ksize=3)
        mouth_edges = cv2.Sobel(mouth_region, cv2.CV_64F, 1, 0, ksize=3)
        
        forehead_wrinkle_score = np.mean(np.absolute(forehead_edges))
        eye_wrinkle_score = np.mean(np.absolute(eye_edges))
        mouth_wrinkle_score = np.mean(np.absolute(mouth_edges))
        
        # 3. Color variance analysis (skin tone evenness decreases with age)
        hsv_face = cv2.cvtColor(face_image, cv2.COLOR_RGB2HSV)
        skin_tone_variance = np.var(hsv_face[:,:,1])  # Variance in saturation channel
        
        # 4. Apply age corrections based on detailed analysis with smaller factors
        # Store original age for comparison
        original_age = age
        
        # 4.1 Face shape correction (more refined) - LIMIT IMPACT of face shape
        if face_size_ratio > 0.95:  # Rounder face
            if age > 30:
                # Rounder faces appear younger in adults - reduced impact
                age_adjustment = -0.02 * (age - 30)  # Reduced impact
                age = age + age_adjustment
        elif face_size_ratio < 0.85:  # More elongated face
            if age < 40:
                # Elongated faces appear older in younger people - reduced impact
                age_adjustment = 0.02 * (40 - age)  # Reduced impact
                age = age + age_adjustment
        
        # 4.2 Wrinkle-based corrections - GREATLY REDUCE IMPACT
        # Forehead wrinkles are significant for age
        if forehead_wrinkle_score > 18:  # Increased threshold from 15 to 18
            if age < 45:
                age += (forehead_wrinkle_score - 18) * 0.08  # Reduced impact
        elif forehead_wrinkle_score < 8 and age > 30:
            age -= (8 - forehead_wrinkle_score) * 0.03  # Reduced impact
        
        # Eye wrinkles (crow's feet)
        if eye_wrinkle_score > 20:  # Increased threshold from 18 to 20
            if age < 50:
                age += (eye_wrinkle_score - 20) * 0.06  # Reduced impact
        
        # Mouth wrinkles (nasolabial folds)
        if mouth_wrinkle_score > 22:  # Increased threshold from 20 to 22
            if age < 55:
                age += (mouth_wrinkle_score - 22) * 0.07  # Reduced impact
        
        # 4.3 Texture variance indicates skin smoothness - REDUCE IMPACT
        if texture_variance < 800 and age > 40:
            # Smoother skin for reported age, reduce age
            age_adjustment = min(2, (800 - texture_variance) * 0.003)  # Reduced impact
            age -= age_adjustment
        elif texture_variance > 1500 and age < 50:
            # More textured skin for reported age, increase age
            age_adjustment = min(3, (texture_variance - 1500) * 0.003)  # Reduced impact
            age += age_adjustment
        
        # 4.4 Skin tone evenness - REDUCE IMPACT
        if skin_tone_variance < 100 and age > 35:
            # Even skin tone suggests younger appearance
            age -= min(1.5, (100 - skin_tone_variance) * 0.008)  # Reduced impact
        elif skin_tone_variance > 200 and age < 45:
            # Uneven skin tone suggests older appearance
            age += min(2, (skin_tone_variance - 200) * 0.004)  # Reduced impact
        
        # Check if age has changed too much from initial estimate
        if abs(age - original_age) > original_age * 0.4:  # If change >40%
            # Adjust back using weighted average
            age = original_age * 0.6 + age * 0.4  # Favor original age more
        
        # 5. Improved confidence-based regression toward mean
        # Apply different regression models for different age ranges
        if age < 10:
            # Children
            regression_strength = 0.1  # Reduced from 0.15
            mean_age = 8
            age = age * (1 - regression_strength) + mean_age * regression_strength
        elif age < 20:
            # Teens
            regression_strength = 0.08  # Reduced from 0.1
            mean_age = 16
            age = age * (1 - regression_strength) + mean_age * regression_strength
        elif age > 75:
            # Elderly
            regression_strength = 0.08  # Reduced from 0.1
            mean_age = 80
            age = age * (1 - regression_strength) + mean_age * regression_strength
        
        # 6. Apply calibration factors if available
        if age_calibration['calibrated']:
            # Limit calibration adjustment effect
            calibrated_age = age * age_calibration['multiplier'] + age_calibration['offset']
            # Only apply 70% of calibration effect to avoid drifting too far
            age = age * 0.3 + calibrated_age * 0.7
            
            # Apply specific group calibration with limited impact
            age_group = categorize_age(age)
            if age_group == 'child':
                factor = max(0.8, min(1.2, age_calibration['child_factor']))  # Limit factor
                age = age * factor
            elif age_group == 'teen':
                factor = max(0.8, min(1.2, age_calibration['teen_factor']))  # Limit factor
                age = age * factor
            elif age_group == 'adult':
                factor = max(0.9, min(1.1, age_calibration['adult_factor']))  # Limit factor
                age = age * factor
            elif age_group == 'senior':
                factor = max(0.9, min(1.1, age_calibration['senior_factor']))  # Limit factor
                age = age * factor
    
    except Exception as e:
        print(f"Error in age estimation adjustments: {e}")
        # If error occurs in adjustments, keep original age
        pass
    
    # 7. Final bounds checking and rounding
    # Ensure age is always within reasonable range
    age = max(1, min(100, age))  # Minimum age 1, maximum 100
    
    # Add smoothing algorithm based on age group
    age_group = categorize_age(age)
    if age_group == 'child':
        # Children usually have more exact ages (e.g., 3, 4, 5...)
        # Round to nearest 0.5 then round up/down
        age = round(round(age * 2) / 2)
    elif age_group == 'teen' or age_group == 'adult':
        # Round to nearest integer
        age = round(age)
    else:  # senior
        # Older people often round to nearest 5 years (e.g., 65, 70, 75)
        # But we still keep higher precision
        age = round(age)
        
    # Print final debug info
    print(f"Final adjusted age: {age}")
    
    return age 