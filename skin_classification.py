import cv2
import numpy as np
import math

def classify_skin_color(face_image):
    """
    Classify skin color from a face image
    
    Args:
        face_image: Input face image
    
    Returns:
        tuple: (skin_color, confidence) - detected skin color type and confidence level
    """
    # Input validation
    if face_image is None or face_image.size == 0:
        print("Invalid image in skin classification, using default (White)")
        return "White", 0.7  # Changed default from Yellow to White
    
    try:
        # Ensure image is in RGB format
        if len(face_image.shape) < 3:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
        
        # Normalize image size
        face_image = cv2.resize(face_image, (224, 224))
        
        # Convert to different color spaces for better skin detection
        hsv_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
        ycrcb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2YCrCb)
        lab_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
        
        # Extract skin regions by masking non-skin pixels
        # HSV skin thresholds - adjusted for better detection
        lower_hsv = np.array([0, 10, 40], dtype=np.uint8)  # Reduced saturation minimum to capture lighter skin
        upper_hsv = np.array([30, 170, 255], dtype=np.uint8)  # Increased value maximum
        hsv_mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
        
        # YCrCb skin thresholds - adjusted for better detection of white skin
        lower_ycrcb = np.array([80, 133, 77], dtype=np.uint8)  # Adjusted to better detect white skin
        upper_ycrcb = np.array([240, 173, 127], dtype=np.uint8)  # Adjusted to better detect white skin
        ycrcb_mask = cv2.inRange(ycrcb_image, lower_ycrcb, upper_ycrcb)
        
        # Combine masks to get better skin segmentation
        skin_mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)
        
        # Apply the mask to extract skin regions
        skin_region = cv2.bitwise_and(face_image, face_image, mask=skin_mask)
        
        # Print debug info
        print("Skin detection analysis:")
        
        # Calculate average color of skin region
        total_pixels = np.count_nonzero(skin_mask)
        if total_pixels > 0:
            # BGR channels
            b_channel, g_channel, r_channel = cv2.split(skin_region)
            b_avg = np.sum(b_channel) / total_pixels
            g_avg = np.sum(g_channel) / total_pixels
            r_avg = np.sum(r_channel) / total_pixels
            
            # HSV stats for skin region
            h_channel, s_channel, v_channel = cv2.split(cv2.bitwise_and(hsv_image, hsv_image, mask=skin_mask))
            h_avg = np.sum(h_channel) / total_pixels
            s_avg = np.sum(s_channel) / total_pixels
            v_avg = np.sum(v_channel) / total_pixels
            
            # LAB stats for skin region
            l_channel, a_channel, b_channel = cv2.split(cv2.bitwise_and(lab_image, lab_image, mask=skin_mask))
            l_avg = np.sum(l_channel) / total_pixels
            a_avg = np.sum(a_channel) / total_pixels
            b_avg_lab = np.sum(b_channel) / total_pixels
        else:
            # Default values if no skin detected - adjusted to favor white skin
            b_avg, g_avg, r_avg = 170, 165, 180  # Higher RGB values for lighter skin
            h_avg, s_avg, v_avg = 15, 80, 180  # Lower saturation, higher value
            l_avg, a_avg, b_avg_lab = 180, 128, 120  # Higher L, lower b for white skin
        
        # Print color metrics for debugging
        print(f"- BGR: ({b_avg:.1f}, {g_avg:.1f}, {r_avg:.1f})")
        print(f"- HSV: ({h_avg:.1f}, {s_avg:.1f}, {v_avg:.1f})") 
        print(f"- LAB: ({l_avg:.1f}, {a_avg:.1f}, {b_avg_lab:.1f})")
        
        # Calculate brightness
        brightness = 0.299 * r_avg + 0.587 * g_avg + 0.114 * b_avg
        
        # Calculate ratios
        rb_ratio = r_avg / (b_avg + 1e-6)  # Avoid division by zero
        rg_ratio = r_avg / (g_avg + 1e-6)
        gb_ratio = g_avg / (b_avg + 1e-6)
        
        # Print additional metrics
        print(f"- Brightness: {brightness:.1f}")
        print(f"- R/B ratio: {rb_ratio:.2f}")
        print(f"- R/G ratio: {rg_ratio:.2f}")
        print(f"- G/B ratio: {gb_ratio:.2f}")
        
        # Calculate scores for each skin type
        white_score = 0.0
        black_score = 0.0
        yellow_score = 0.0
        
        # Brightness rules (adjusted to favor white over yellow)
        if brightness > 180:  # Very bright - likely white skin
            white_score += 0.8  # Increased from 0.7
            yellow_score += 0.1  # Decreased from 0.2
            black_score += 0.1
            print("- High brightness indicates White skin (+0.8)")
        elif brightness > 160:  # Moderately bright - could be white or yellow
            white_score += 0.6  # Increased from 0.5
            yellow_score += 0.3  # Decreased from 0.4
            black_score += 0.1
            print("- Medium-high brightness: White (+0.6), Yellow (+0.3)")
        elif brightness > 120:  # Medium brightness - could be yellow or white
            yellow_score += 0.5  # Decreased from 0.6
            white_score += 0.4  # Increased from 0.3
            black_score += 0.1
            print("- Medium brightness: Yellow (+0.5), White (+0.4)")
        elif brightness < 100:  # Dark - likely black skin
            black_score += 0.7
            yellow_score += 0.2
            white_score += 0.1
            print("- Low brightness indicates Black skin (+0.7)")
        else:  # Medium-low brightness
            yellow_score += 0.4
            black_score += 0.4
            white_score += 0.2
            print("- Medium-low brightness: Yellow (+0.4), Black (+0.4)")
        
        # Red-Blue ratio rules (adjusted for better discrimination)
        if rb_ratio > 1.5:
            if rb_ratio > 1.7:  # Very high R/B ratio - could be yellow skin
                yellow_score += 0.5  # Decreased from 0.6
                white_score += 0.4  # Increased from 0.3
                black_score += 0.1
                print(f"- Very high R/B ratio ({rb_ratio:.2f}): Yellow (+0.5), White (+0.4)")
            else:  # Moderate high R/B ratio
                yellow_score += 0.4
                white_score += 0.4
                black_score += 0.2
                print(f"- High R/B ratio ({rb_ratio:.2f}): Yellow (+0.4), White (+0.4)")
        elif rb_ratio < 1.2:  # Low R/B ratio
            white_score += 0.6  # Increased from 0.5
            black_score += 0.2  # Decreased from 0.3
            yellow_score += 0.2
            print(f"- Low R/B ratio ({rb_ratio:.2f}) indicates White skin (+0.6)")
        
        # Red-Green ratio rules (adjusted)
        if rg_ratio > 1.25:  # High R/G ratio
            yellow_score += 0.3  # Decreased from 0.4
            white_score += 0.3  # Increased from 0.2
            print(f"- High R/G ratio ({rg_ratio:.2f}): Yellow (+0.3), White (+0.3)")
        elif rg_ratio < 1.12:  # Low R/G ratio
            white_score += 0.5  # Increased from 0.4
            black_score += 0.2
            print(f"- Low R/G ratio ({rg_ratio:.2f}) indicates White skin (+0.5)")
        
        # LAB space for discrimination (adjusted thresholds)
        # b in LAB: positive = yellow, negative = blue
        if b_avg_lab > 145:  # High yellow component in LAB
            yellow_score += 0.5
            white_score += 0.2
            print(f"- Very high b value in LAB ({b_avg_lab:.1f}) indicates Yellow skin (+0.5)")
        elif b_avg_lab > 135:  # Moderate yellow component in LAB
            yellow_score += 0.4
            white_score += 0.3
            print(f"- High b value in LAB ({b_avg_lab:.1f}): Yellow (+0.4), White (+0.3)")
        elif b_avg_lab < 128:  # Low yellow component in LAB - adjusted threshold
            white_score += 0.6  # Increased from 0.5
            yellow_score += 0.1  # Decreased from 0.2
            print(f"- Low b value in LAB ({b_avg_lab:.1f}) indicates White skin (+0.6)")
        
        # HSV saturation (adjusted)
        if s_avg > 45:  # High saturation - adjusted threshold up
            yellow_score += 0.4
            black_score += 0.2
            print(f"- High saturation ({s_avg:.1f}) indicates Yellow skin (+0.4)")
        elif s_avg < 30:  # Low saturation - adjusted threshold
            white_score += 0.5  # Increased from 0.4
            print(f"- Low saturation ({s_avg:.1f}) indicates White skin (+0.5)")
        else:  # Medium saturation
            white_score += 0.3
            yellow_score += 0.3
            print(f"- Medium saturation ({s_avg:.1f}): White (+0.3), Yellow (+0.3)")
        
        # Hue from HSV (adjusted)
        if 12 <= h_avg <= 20:  # Typical yellow tone
            yellow_score += 0.4
            print(f"- Typical yellow hue ({h_avg:.1f}) indicates Yellow skin (+0.4)")
        elif h_avg < 10:  # More reddish tone
            white_score += 0.4  # Increased from 0.3
            print(f"- Reddish hue ({h_avg:.1f}) indicates White skin (+0.4)")
        elif h_avg < 12:  # Borderline case
            white_score += 0.3
            yellow_score += 0.2
            print(f"- Borderline hue ({h_avg:.1f}): White (+0.3), Yellow (+0.2)")
        
        # L channel in LAB (new feature)
        if l_avg > 160:  # Very high lightness
            white_score += 0.5
            print(f"- Very high lightness (L={l_avg:.1f}) indicates White skin (+0.5)")
        elif l_avg > 140:  # High lightness
            white_score += 0.4
            yellow_score += 0.2
            print(f"- High lightness (L={l_avg:.1f}): White (+0.4), Yellow (+0.2)")
        
        # Normalize scores
        total_score = white_score + black_score + yellow_score
        white_score = white_score / total_score
        black_score = black_score / total_score
        yellow_score = yellow_score / total_score
        
        # Determine skin color type with highest score
        scores = {
            "White": white_score,
            "Black": black_score,
            "Yellow": yellow_score
        }
        
        # Print normalized scores
        print(f"Normalized scores: White: {white_score:.2f}, Black: {black_score:.2f}, Yellow: {yellow_score:.2f}")
        
        # Find the skin color with the highest score
        skin_color = max(scores, key=scores.get)
        confidence = scores[skin_color]
        
        # Apply correction for borderline cases (enhanced)
        # If scores are close between white and yellow, consider additional features
        if skin_color == "Yellow" and white_score > 0.35 and (yellow_score - white_score) < 0.15:  # Increased threshold
            # If brightness is high or saturation is low, prioritize white
            if brightness > 170 or s_avg < 30 or l_avg > 150:
                skin_color = "White"
                confidence = white_score
                print("Correction applied: Changed Yellow to White due to high brightness, high lightness, or low saturation")
        
        # Less aggressive correction from White to Yellow
        if skin_color == "White" and yellow_score > 0.40 and (white_score - yellow_score) < 0.08:
            # Only if b value in LAB is very high and saturation is also high
            if b_avg_lab > 145 and s_avg > 50:
                skin_color = "Yellow"
                confidence = yellow_score
                print("Correction applied: Changed White to Yellow due to very high b value and high saturation")
        
        print(f"Detected skin color: {skin_color} (confidence: {confidence:.2f})")
        return skin_color, confidence
        
    except Exception as e:
        print(f"Error in skin color classification: {e}")
        # Return default value on error - changed to White
        return "White", 0.7

def get_skin_vector(face_image, vector_length=16):
    """
    Create feature vector for skin color from face image
    
    Args:
        face_image: Face image
        vector_length: Length of output vector
        
    Returns:
        ndarray: Skin color feature vector (16-dimensional)
    """
    # Initialize feature vector with float Python
    skin_vector = [0.0] * vector_length
    
    # Validate input
    if face_image is None or face_image.size == 0:
        return np.array(skin_vector, dtype=float)
    
    try:
        # Detect skin color and confidence
        skin_color, confidence = classify_skin_color(face_image)
        confidence = float(confidence)
        
        # Define color indices for one-hot encoding
        color_mapping = {
            "White": 0,
            "Black": 1,
            "Yellow": 2,
            "Unknown": 3
        }
        color_index = color_mapping.get(skin_color, 3)  # Default to unknown
        
        # Create a copy for analysis
        if len(face_image.shape) == 3:
            face_copy = face_image.copy()
        else:
            face_copy = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
        
        # Resize for consistency
        face_copy = cv2.resize(face_copy, (128, 128))
        
        # Convert to different color spaces
        hsv_image = cv2.cvtColor(face_copy, cv2.COLOR_BGR2HSV)
        lab_image = cv2.cvtColor(face_copy, cv2.COLOR_BGR2LAB)
        
        # Extract skin mask - adjusted for better white skin detection
        lower_hsv = np.array([0, 10, 40], dtype=np.uint8)  # Adjusted thresholds
        upper_hsv = np.array([30, 170, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
        
        # Apply mask to get skin region
        skin_region = cv2.bitwise_and(face_copy, face_copy, mask=skin_mask)
        
        # Calculate skin region statistics
        total_skin_pixels = np.count_nonzero(skin_mask)
        if total_skin_pixels > 0:
            # BGR channels
            b, g, r = cv2.split(skin_region)
            # HSV channels
            h, s, v = cv2.split(cv2.bitwise_and(hsv_image, hsv_image, mask=skin_mask))
            # LAB channels
            l, a, bb = cv2.split(cv2.bitwise_and(lab_image, lab_image, mask=skin_mask))
            
            # Calculate statistics only on skin pixels
            r_mean = float(np.sum(r) / total_skin_pixels / 255.0)
            g_mean = float(np.sum(g) / total_skin_pixels / 255.0)
            b_mean = float(np.sum(b) / total_skin_pixels / 255.0)
            
            h_mean = float(np.sum(h * (skin_mask > 0)) / total_skin_pixels / 180.0)
            s_mean = float(np.sum(s * (skin_mask > 0)) / total_skin_pixels / 255.0)
            v_mean = float(np.sum(v * (skin_mask > 0)) / total_skin_pixels / 255.0)
            
            # LAB features
            l_mean = float(np.sum(l * (skin_mask > 0)) / total_skin_pixels / 255.0)
            a_mean = float(np.sum(a * (skin_mask > 0)) / total_skin_pixels / 255.0)
            b_mean_lab = float(np.sum(bb * (skin_mask > 0)) / total_skin_pixels / 255.0)
        else:
            # Default values if no skin detected - adjusted to favor white skin
            r_mean, g_mean, b_mean = 0.7, 0.65, 0.65  # Higher RGB for lighter skin
            h_mean, s_mean, v_mean = 0.05, 0.3, 0.7  # Lower saturation, higher value
            l_mean, a_mean, b_mean_lab = 0.7, 0.5, 0.45  # Higher L, lower b for white skin
        
        # Fill vector with skin-specific features
        skin_vector[0] = float(r_mean)  # Red channel mean
        skin_vector[1] = float(g_mean)  # Green channel mean
        skin_vector[2] = float(b_mean)  # Blue channel mean
        skin_vector[3] = float(h_mean)  # Hue mean
        skin_vector[4] = float(s_mean)  # Saturation mean
        skin_vector[5] = float(v_mean)  # Value mean
        skin_vector[6] = float(confidence)  # Confidence
        
        # Calculate brightness and ratios
        brightness = float(0.299 * r_mean + 0.587 * g_mean + 0.114 * b_mean)
        rb_ratio = float(r_mean / (b_mean + 1e-6))
        rg_ratio = float(r_mean / (g_mean + 1e-6))
        
        skin_vector[7] = float(brightness)
        skin_vector[8] = float(rb_ratio)
        skin_vector[9] = float(rg_ratio)
        
        # LAB color space features
        skin_vector[10] = float(l_mean)  # Lightness (L)
        skin_vector[11] = float(b_mean_lab)  # Yellow-Blue axis (b)
        
        # One-hot encoding for skin color type
        # Use positions 12-15 (different from other vectors)
        skin_vector[12 + color_index] = 1.0
        
    except Exception as e:
        print(f"Error creating skin color vector: {e}")
    
    # Convert any potential NaN values to 0
    for i in range(len(skin_vector)):
        if math.isnan(float(skin_vector[i])):
            skin_vector[i] = 0.0
        # Ensure all values are Python float
        skin_vector[i] = float(skin_vector[i])
    
    # Return as NumPy array with float dtype
    return np.array(skin_vector, dtype=float)