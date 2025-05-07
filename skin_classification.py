import numpy as np
import cv2

def classify_skin_color(hsv_face):
    """Classify skin color into White, Yellow, or Black based on HSV values"""
    try:
        # Check input image
        if hsv_face is None or hsv_face.size == 0:
            return "unknown"
        
        # Ensure HSV format
        if len(hsv_face.shape) < 3 or hsv_face.shape[2] != 3:
            return "unknown"
        
        # Apply Gaussian blur for smoothing before analysis
        smoothed_face = cv2.GaussianBlur(hsv_face, (5, 5), 0)
        
        # Extend the skin color range to handle special cases
        lower_skin = np.array([0, 15, 50], dtype=np.uint8)
        upper_skin = np.array([50, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(smoothed_face, lower_skin, upper_skin)
        
        # Apply morphological operations to improve the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Extract skin pixels
        skin_pixels = smoothed_face[mask > 0]
        
        # If not enough skin pixels, try again with a lower threshold
        if skin_pixels.size < 100:
            lower_skin = np.array([0, 10, 40], dtype=np.uint8)
            upper_skin = np.array([60, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(smoothed_face, lower_skin, upper_skin)
            skin_pixels = smoothed_face[mask > 0]
        
        # If still not enough, use the entire image and estimate probability
        if skin_pixels.size < 100:
            # Use center pixels of the image (assume it's the face region)
            center_region = hsv_face[hsv_face.shape[0]//4:3*hsv_face.shape[0]//4, 
                              hsv_face.shape[1]//4:3*hsv_face.shape[1]//4]
            if center_region.size > 0:
                skin_pixels = center_region.reshape(-1, 3)
            else:
                skin_pixels = hsv_face.reshape(-1, 3)
        
        # Calculate average values for HSV channels
        avg_hue = np.mean(skin_pixels[:, 0])
        avg_saturation = np.mean(skin_pixels[:, 1])
        avg_value = np.mean(skin_pixels[:, 2])
        
        # Calculate standard deviation to assess reliability
        std_hue = np.std(skin_pixels[:, 0])
        std_saturation = np.std(skin_pixels[:, 1])
        std_value = np.std(skin_pixels[:, 2])
        
        # Evaluate reliability based on standard deviation
        reliability = 1.0
        if std_hue > 20 or std_saturation > 60 or std_value > 60:
            reliability = 0.7  # Reduce reliability if there's high variation
        
        # Classification based on adjusted thresholds
        # White skin: High value, low saturation
        if avg_value > 170 and avg_saturation < 90:
            return "White"
            
        # Black skin: Low value
        elif avg_value < 130:
            return "Black"
            
        # Yellow skin: Yellow hue, medium value, medium saturation
        elif 5 <= avg_hue <= 30 and avg_saturation > 30:
            return "Yellow"
            
        # Remaining cases
        else:
            # Use fallback rules based on ratio and reliability
            if avg_value > 150:
                if avg_hue > 10 and avg_hue < 25 and avg_saturation > 40:
                    return "Yellow"
                else:
                    return "White"
            elif avg_value < 110:
                return "Black"
            else:
                # Determine skin color based on combination of factors
                if avg_saturation > 80:
                    if avg_hue > 10 and avg_hue < 25:
                        return "Yellow"
                    else:
                        return "Black"
                else:
                    return "White"
    except Exception as e:
        print(f"Error in skin color classification: {e}")
        return "unknown" 