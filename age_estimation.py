import cv2
import numpy as np
import os
import re
from deepface import DeepFace

def extract_age_from_filename_robust(filename):
    """
    Enhanced method to extract age from filename
    Supporting various filename formats
    """
    # Standard format: 35_0_0_20170104183524965.jpg
    try:
        # Method 1: Extract first number before underscore
        parts = filename.split('_')
        if len(parts) >= 1 and parts[0].isdigit():
            return int(parts[0])
        
        # Method 2: Search for age pattern in filename
        age_pattern = r'(\d+)_\d+_\d+_\d+'
        match = re.search(age_pattern, filename)
        if match:
            return int(match.group(1))
        
        # Method 3: Try finding any 2-digit number at start of filename
        match = re.match(r'^(\d{1,2}).*', filename)
        if match:
            age = int(match.group(1))
            if 32 <= age <= 53:  # Only accept ages in our target range
                return age
    except Exception as e:
        print(f"Error extracting age from filename {filename}: {e}")
    
    return None

def calibrate_age_estimation(data_folder):
    """
    Specialized calibration function focused on 32-53 age range.
    Analyzes facial features specific to this age range.
    """
    print("Starting specialized age estimation calibration for ages 32-53...")
    
    # Check if folder exists
    if not os.path.exists(data_folder):
        print(f"Error: Data folder '{data_folder}' not found")
        return
    
    try:
        # Find images with valid age in filename
        image_files = [f for f in os.listdir(data_folder) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        calibration_images = []
        for img_file in image_files:
            age = extract_age_from_filename_robust(img_file)
            if age is not None and 32 <= age <= 53:
                calibration_images.append((img_file, age))
        
        if len(calibration_images) < 5:
            print(f"Warning: Only {len(calibration_images)} images with valid age (32-53) found. Calibration may not be accurate.")
            return
            
        print(f"Found {len(calibration_images)} images with age 32-53 for calibration")
        
        # Analysis for specific age sub-ranges
        age_ranges = {
            "32-37": {"count": 0, "texture": [], "wrinkle": [], "edge": []},
            "38-43": {"count": 0, "texture": [], "wrinkle": [], "edge": []},
            "44-48": {"count": 0, "texture": [], "wrinkle": [], "edge": []},
            "49-53": {"count": 0, "texture": [], "wrinkle": [], "edge": []}
        }
        
        # Analyze a subset of images for performance
        sample_size = min(100, len(calibration_images))
        for i in range(sample_size):
            img_file, age = calibration_images[i]
            img_path = os.path.join(data_folder, img_file)
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Analyze features
            features = analyze_facial_features(img)
            if features is None:
                continue
                
            # Assign to age range
            if 32 <= age <= 37:
                age_range = "32-37"
            elif 38 <= age <= 43:
                age_range = "38-43"
            elif 44 <= age <= 48:
                age_range = "44-48"
            else:
                age_range = "49-53"
                
            # Add features to age range
            age_ranges[age_range]["count"] += 1
            age_ranges[age_range]["texture"].append(features["texture_variance"])
            age_ranges[age_range]["wrinkle"].append(features["forehead_edge_intensity"] + features["eyes_edge_intensity"])
            age_ranges[age_range]["edge"].append(features["edge_intensity"])
        
        # Calculate averages and log the results
        for age_range, data in age_ranges.items():
            if data["count"] > 0:
                avg_texture = sum(data["texture"]) / data["count"] if data["count"] > 0 else 0
                avg_wrinkle = sum(data["wrinkle"]) / data["count"] if data["count"] > 0 else 0
                avg_edge = sum(data["edge"]) / data["count"] if data["count"] > 0 else 0
                print(f"Age range {age_range}: {data['count']} images, avg texture: {avg_texture:.2f}, avg wrinkle: {avg_wrinkle:.4f}, avg edge: {avg_edge:.4f}")
            else:
                print(f"Age range {age_range}: No images found")
        
        print("Age estimation calibration completed")
        
    except Exception as e:
        print(f"Error during age estimation calibration: {e}")

def analyze_facial_features(face_image):
    """
    Analyze facial features with a focus on characteristics of 32-53 age range
    """
    if face_image is None or face_image.size == 0:
        return None
    
    try:
        # Convert to grayscale if needed
        if len(face_image.shape) == 3:
            gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_image.copy()
        
        height, width = gray_face.shape[:2]
        face_ratio = width / height
        
        # Apply filter to reduce noise
        gray_face = cv2.GaussianBlur(gray_face, (5, 5), 0)
        
        # Analyze skin smoothness using Laplacian for texture analysis
        lap_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        
        # Analyze edge intensity in the face for wrinkle detection
        edges = cv2.Canny(gray_face, 50, 150)
        edge_intensity = np.sum(edges) / (width * height)
        
        # Analyze contrast in the face
        contrast = np.std(gray_face)
        
        # Additional analysis by facial regions (focused on aging features)
        # Forehead region (top third) - wrinkles appear here early
        forehead = gray_face[0:int(height/3), :]
        forehead_texture = cv2.Laplacian(forehead, cv2.CV_64F).var()
        
        # Eye region (second third) - crow's feet and eye bags
        eyes = gray_face[int(height/3):int(2*height/3), :]
        eyes_texture = cv2.Laplacian(eyes, cv2.CV_64F).var()
        
        # Mouth region (bottom third) - nasolabial folds and marionette lines
        mouth = gray_face[int(2*height/3):, :]
        mouth_texture = cv2.Laplacian(mouth, cv2.CV_64F).var()
        
        # Analyze edges in each region (wrinkles)
        forehead_edges = cv2.Canny(forehead, 50, 150)
        forehead_edge_intensity = np.sum(forehead_edges) / (forehead.shape[0] * forehead.shape[1]) if forehead.size > 0 else 0
        
        eyes_edges = cv2.Canny(eyes, 50, 150)
        eyes_edge_intensity = np.sum(eyes_edges) / (eyes.shape[0] * eyes.shape[1]) if eyes.size > 0 else 0
        
        mouth_edges = cv2.Canny(mouth, 50, 150)
        mouth_edge_intensity = np.sum(mouth_edges) / (mouth.shape[0] * mouth.shape[1]) if mouth.size > 0 else 0
        
        # Analyze horizontal edges specifically (wrinkles tend to be horizontal)
        sobelx = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
        horizontal_edge_intensity = np.mean(np.absolute(sobelx))
        
        # Analyze nasolabial fold region (specific to aging 35+)
        # This region runs from sides of nose to corners of mouth
        mid_y = int(height/2)
        mid_x = int(width/2)
        
        # Left and right nasolabial regions (approximated)
        left_nasolabial = gray_face[mid_y:int(3*height/4), int(width/4):mid_x]
        right_nasolabial = gray_face[mid_y:int(3*height/4), mid_x:int(3*width/4)]
        
        # Calculate texture and edges in nasolabial regions
        left_nl_texture = cv2.Laplacian(left_nasolabial, cv2.CV_64F).var() if left_nasolabial.size > 0 else 0
        right_nl_texture = cv2.Laplacian(right_nasolabial, cv2.CV_64F).var() if right_nasolabial.size > 0 else 0
        nasolabial_texture = (left_nl_texture + right_nl_texture) / 2 if left_nasolabial.size > 0 and right_nasolabial.size > 0 else 0
        
        # Return comprehensive features dictionary
        features = {
            'face_ratio': face_ratio,
            'texture_variance': lap_var,
            'edge_intensity': edge_intensity,
            'contrast': contrast,
            'forehead_texture': forehead_texture,
            'eyes_texture': eyes_texture,
            'mouth_texture': mouth_texture,
            'forehead_edge_intensity': forehead_edge_intensity,
            'eyes_edge_intensity': eyes_edge_intensity,
            'mouth_edge_intensity': mouth_edge_intensity,
            'horizontal_edge_intensity': horizontal_edge_intensity,
            'nasolabial_texture': nasolabial_texture
        }
        
        return features
        
    except Exception as e:
        print(f"Error analyzing facial features: {e}")
        return None

def calculate_adult_age_confidence(features, estimated_age):
    """
    Calculate confidence in age estimation based on facial features
    Specialized for 32-53 age range
    """
    # Start with medium confidence
    confidence = 0.7
    
    # Adjust confidence based on feature intensity for specific ages
    if estimated_age < 38:
        # Early 30s
        if features['texture_variance'] < 180 and features['edge_intensity'] < 0.14:
            confidence += 0.15  # Clear early-30s features
        elif features['nasolabial_texture'] < 180:
            confidence += 0.1   # Less pronounced nasolabial folds
    elif 38 <= estimated_age <= 45:
        # Late 30s to mid 40s
        if 180 <= features['texture_variance'] <= 230 and 0.14 <= features['edge_intensity'] <= 0.17:
            confidence += 0.15  # Clear early-40s features
        elif 180 <= features['nasolabial_texture'] <= 230:
            confidence += 0.1   # Moderate nasolabial folds
    elif estimated_age > 45:
        # Late 40s and early 50s
        if features['texture_variance'] > 230 and features['edge_intensity'] > 0.17:
            confidence += 0.15  # Very clear late-40s and early-50s features
        elif features['nasolabial_texture'] > 230:
            confidence += 0.1   # Pronounced nasolabial folds
    
    # Adjust based on specific wrinkle patterns for accurate age determination
    if estimated_age < 40 and (features['forehead_edge_intensity'] > 0.18 or features['eyes_edge_intensity'] > 0.18):
        confidence -= 0.1  # Unusual to have strong wrinkles in 30s
    elif estimated_age > 45 and features['forehead_edge_intensity'] < 0.14 and features['eyes_edge_intensity'] < 0.14:
        confidence -= 0.1  # Unusual to have few wrinkles in late 40s+
        
    # Limit range
    return max(0.5, min(0.95, confidence))

def adjust_age_estimation(age, face_image):
    """
    Specialized age estimation for adults 32-53 years old
    Focuses on features specific to this age range
    """
    # Check for valid input
    if face_image is None or face_image.size == 0:
        print("Warning: Empty or invalid face image")
        return max(32, min(53, age))
    
    # Initial age from DeepFace - limit to our range
    initial_age = max(32, min(53, age))
    print(f"Initial DeepFace age: {initial_age}")
    
    try:
        # 1. ANALYZE FACIAL FEATURES
        features = analyze_facial_features(face_image)
        if features is None:
            print("Could not analyze facial features")
            return initial_age

        print(f"Face analysis - texture: {features['texture_variance']:.2f}, edges: {features['edge_intensity']:.4f}, forehead: {features['forehead_edge_intensity']:.4f}, eyes: {features['eyes_edge_intensity']:.4f}")
        
        # Define confidence in DeepFace prediction
        deepface_confidence = 0.7  # Default confidence
        
        # 2. ADJUST DEEPFACE CONFIDENCE BASED ON AGE
        # Lower confidence for boundary ages (near 32 or 53) as DeepFace tends to struggle at boundaries
        if initial_age < 34 or initial_age > 51:
            deepface_confidence = 0.65
        # Higher confidence for middle of our range where DeepFace performs better
        elif 38 <= initial_age <= 47:
            deepface_confidence = 0.8
            
        # 3. ANALYZE AGE BASED ON FACIAL FEATURES
        
        # Start with weighted base calculation using textural features
        normalized_texture = min(1.0, features['texture_variance'] / 350)
        normalized_edge = min(1.0, features['edge_intensity'] / 0.25)
        normalized_forehead = min(1.0, features['forehead_edge_intensity'] / 0.25)
        normalized_eyes = min(1.0, features['eyes_edge_intensity'] / 0.25)
        normalized_nasolabial = min(1.0, features['nasolabial_texture'] / 350)
        
        # Calculate early-mid 30s probability (32-37)
        early30s_score = (
            (1 - normalized_texture) * 0.25 +
            (1 - normalized_edge) * 0.2 +
            (1 - normalized_forehead) * 0.2 +
            (1 - normalized_eyes) * 0.2 +
            (1 - normalized_nasolabial) * 0.15
        )
        
        # Calculate late 30s-early 40s probability (38-43)
        early40s_score = (
            (0.5 - abs(normalized_texture - 0.5)) * 0.3 +
            (0.5 - abs(normalized_edge - 0.5)) * 0.2 +
            (0.5 - abs(normalized_forehead - 0.5)) * 0.2 +
            (0.5 - abs(normalized_eyes - 0.5)) * 0.15 +
            (0.5 - abs(normalized_nasolabial - 0.5)) * 0.15
        ) * 2  # Scale to 0-1
        
        # Calculate mid-late 40s probability (44-48)
        mid40s_score = (
            (0.75 - abs(normalized_texture - 0.75)) * 0.25 +
            (0.7 - abs(normalized_edge - 0.7)) * 0.2 +
            (0.7 - abs(normalized_forehead - 0.7)) * 0.2 +
            (0.7 - abs(normalized_eyes - 0.7)) * 0.2 +
            (0.7 - abs(normalized_nasolabial - 0.7)) * 0.15
        ) * 2  # Scale to 0-1
        
        # Calculate early 50s probability (49-53)
        early50s_score = (
            normalized_texture * 0.25 +
            normalized_edge * 0.2 +
            normalized_forehead * 0.2 +
            normalized_eyes * 0.2 +
            normalized_nasolabial * 0.15
        )
        
        # Find highest probability age range
        scores = [
            (early30s_score, 34.5),  # Mid-point of 32-37
            (early40s_score, 40.5),  # Mid-point of 38-43
            (mid40s_score, 46.0),    # Mid-point of 44-48
            (early50s_score, 51.0)   # Mid-point of 49-53
        ]
        
        # Sort by score (descending)
        scores.sort(reverse=True, key=lambda x: x[0])
        
        # Get top two scores for interpolation
        top_score, top_age = scores[0]
        second_score, second_age = scores[1]
        
        # Calculate feature-based age using weighted interpolation between top two ranges
        total_score = top_score + second_score
        if total_score > 0:
            feature_based_age = (top_age * top_score + second_age * second_score) / total_score
        else:
            feature_based_age = top_age
            
        # Ensure feature_based_age is within our range
        feature_based_age = max(32, min(53, feature_based_age))
        
        # Calculate confidence in feature-based estimation
        feature_confidence = calculate_adult_age_confidence(features, feature_based_age)
        print(f"Feature-based age estimate: {feature_based_age:.1f} (confidence: {feature_confidence:.2f})")
        
        # 4. SPECIAL CASES - STRONG INDICATORS
        
        # Very clear indicators of specific age ranges
        if features['texture_variance'] < 150 and features['edge_intensity'] < 0.12 and features['forehead_edge_intensity'] < 0.12:
            # Very smooth skin, few wrinkles - strong indicator of early 30s
            if feature_based_age > 36:
                print("Strong indicators of early 30s detected, adjusting estimate")
                feature_based_age = max(32, min(36, feature_based_age * 0.7 + 33 * 0.3))
                feature_confidence = min(0.9, feature_confidence + 0.1)
                
        elif features['texture_variance'] > 250 and features['edge_intensity'] > 0.18 and features['forehead_edge_intensity'] > 0.18:
            # Very textured skin, significant wrinkles - strong indicator of late 40s/early 50s
            if feature_based_age < 46:
                print("Strong indicators of late 40s/early 50s detected, adjusting estimate")
                feature_based_age = max(46, min(53, feature_based_age * 0.7 + 49 * 0.3))
                feature_confidence = min(0.9, feature_confidence + 0.1)
        
        # 5. RESOLVE CONFLICTS
        age_difference = abs(initial_age - feature_based_age)
        
        # If large difference, adjust confidences
        if age_difference > 8:  # Threshold reduced to be more sensitive in our narrow range
            print(f"CONFLICT: DeepFace ({initial_age}) vs Features ({feature_based_age:.1f})")
            
            # Adjust confidence based on feature strength
            if initial_age < 36 and features['texture_variance'] > 220 and features['forehead_edge_intensity'] > 0.17:
                # DeepFace says early 30s but features suggest late 40s
                deepface_confidence *= 0.7
                print("Reducing DeepFace confidence due to mature features")
            elif initial_age > 48 and features['texture_variance'] < 160 and features['forehead_edge_intensity'] < 0.13:
                # DeepFace says early 50s but features suggest early-mid 30s
                deepface_confidence *= 0.7
                print("Reducing DeepFace confidence due to youthful features")
        
        # 6. COMBINE PREDICTIONS
        # Weights based on confidence
        deepface_weight = deepface_confidence
        feature_weight = feature_confidence
        
        # Normalize weights
        total_weight = deepface_weight + feature_weight
        normalized_deepface_weight = deepface_weight / total_weight
        normalized_feature_weight = feature_weight / total_weight
        
        # Calculate final combined age
        final_age = (initial_age * normalized_deepface_weight) + (feature_based_age * normalized_feature_weight)
        
        print(f"Final combination: DeepFace({initial_age}) * {normalized_deepface_weight:.2f} + Features({feature_based_age:.1f}) * {normalized_feature_weight:.2f} = {final_age:.1f}")
        
        # 7. FINAL REFINEMENTS
        
        # Specific adjustments for nasolabial folds (strong indicator of age in this range)
        if features['nasolabial_texture'] > 240 and final_age < 44:
            correction = min(3.0, (44 - final_age) * 0.4)
            final_age += correction
            print(f"Final correction for pronounced nasolabial folds: +{correction:.1f} years")
        elif features['nasolabial_texture'] < 160 and final_age > 40:
            correction = min(3.0, (final_age - 40) * 0.3)
            final_age -= correction
            print(f"Final correction for minimal nasolabial folds: -{correction:.1f} years")
            
        # Correction for eye region wrinkles (crow's feet - strong indicator in this range)
        if features['eyes_edge_intensity'] > 0.19 and final_age < 45:
            correction = min(2.5, (45 - final_age) * 0.3)
            final_age += correction
            print(f"Final correction for pronounced eye wrinkles: +{correction:.1f} years")
        elif features['eyes_edge_intensity'] < 0.12 and final_age > 38:
            correction = min(2.5, (final_age - 38) * 0.25)
            final_age -= correction
            print(f"Final correction for minimal eye wrinkles: -{correction:.1f} years")
        
        # Ensure age is within our target range
        final_age = max(32, min(53, final_age))
        return round(final_age)
        
    except Exception as e:
        print(f"Error in age estimation: {e}")
        return initial_age  # Return initial age if error occurs 