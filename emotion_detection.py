import cv2
import numpy as np

def improve_emotion_detection(emotion, face_image, emotions_dict):
    """Improve emotion detection by checking consistency and using advanced image analysis"""
    try:
        # Check if face_image is valid
        if face_image is None or face_image.size == 0:
            return "neutral"  # Return default value if image is invalid
            
        # Enhance image contrast for better feature detection
        gray_face = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray_face)
        
        # Check confidence
        max_confidence = max(emotions_dict.values())
        
        # Differentiating value based on image sharpness
        # Calculate sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(enhanced_gray, cv2.CV_64F).var()
        
        # Adjust confidence threshold based on image clarity
        confidence_threshold = 0.35
        if laplacian_var < 100:  # Blurry image
            confidence_threshold = 0.30  # Lower confidence threshold for blurry images
        
        # If confidence is very low, try with other methods
        if max_confidence < confidence_threshold:
            # 1. Face symmetry analysis
            # Split face to check symmetry
            try:
                height, width = face_image.shape[:2]
                left_half = face_image[:, :width//2]
                right_half = face_image[:, width//2:]
                right_half_flipped = cv2.flip(right_half, 1)
                
                # Resize both halves to have the same size
                if left_half.shape != right_half_flipped.shape:
                    min_height = min(left_half.shape[0], right_half_flipped.shape[0])
                    min_width = min(left_half.shape[1], right_half_flipped.shape[1])
                    left_half = left_half[:min_height, :min_width]
                    right_half_flipped = right_half_flipped[:min_height, :min_width]
                
                # Calculate difference between two halves
                diff = np.mean(cv2.absdiff(left_half, right_half_flipped))
                symmetry = 1 - min(1, diff / 50)  # Normalize to 0-1
                
                # Adjust based on symmetry
                if symmetry > 0.8:  # Very symmetrical face
                    if emotions_dict.get('neutral', 0) > 0.2:
                        return 'neutral'
                    elif emotions_dict.get('happy', 0) > 0.2:
                        return 'happy'
                elif symmetry < 0.6:  # Asymmetrical face
                    if emotions_dict.get('surprise', 0) > 0.15:
                        return 'surprise'
                    if emotions_dict.get('disgust', 0) > 0.15:
                        return 'disgust'
                    if emotions_dict.get('sad', 0) > 0.15:
                        return 'sad'
            except Exception as e:
                print(f"Error analyzing face symmetry: {e}")
            
            # 2. Mouth region analysis
            try:
                # Analyze mouth region (bottom 1/3 of face)
                mouth_region = face_image[2*face_image.shape[0]//3:, :]
                mouth_gray = cv2.cvtColor(mouth_region, cv2.COLOR_RGB2GRAY)
                
                # Create binary mask to detect mouth openness
                _, mouth_binary = cv2.threshold(mouth_gray, 60, 255, cv2.THRESH_BINARY)
                
                # Calculate percentage of white pixels (to predict mouth open/closed)
                white_pixel_percentage = np.sum(mouth_binary) / (mouth_binary.size * 255)
                
                # If mouth is open (high white pixel ratio), could be happy, surprise or fear
                if white_pixel_percentage > 0.3:
                    # Determine emotion based on ratio between emotions in emotions_dict
                    smile_emotions = {e: v for e, v in emotions_dict.items() if e in ['happy', 'surprise', 'fear']}
                    if smile_emotions:
                        return max(smile_emotions.items(), key=lambda x: x[1])[0]
            except Exception as e:
                print(f"Error analyzing mouth region: {e}")
            
            # 3. Eye region check
            try:
                # Analyze eye region (middle 1/3 of face)
                eye_region = face_image[face_image.shape[0]//3:2*face_image.shape[0]//3, :]
                eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_RGB2GRAY)
                
                # Use Canny edge detector to detect edges
                edges = cv2.Canny(eye_gray, 50, 150)
                edge_density = np.sum(edges) / edges.size
                
                # High edge density in eye region may indicate negative emotion
                if edge_density > 0.1:
                    negative_emotions = {e: v for e, v in emotions_dict.items() if e in ['anger', 'fear', 'sad']}
                    if negative_emotions:
                        return max(negative_emotions.items(), key=lambda x: x[1])[0]
            except Exception as e:
                print(f"Error analyzing eye region: {e}")
        
        # Compare between two detected emotions with highest confidence
        # If difference is small, choose the more recognizable emotion
        emotions_sorted = sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True)
        if len(emotions_sorted) > 1:
            first, second = emotions_sorted[0], emotions_sorted[1]
            
            # If confidence difference is small
            if first[1] - second[1] < 0.15:
                # Prioritize more easily recognizable emotions (neutral, happy)
                easy_emotions = ['neutral', 'happy']
                if second[0] in easy_emotions:
                    return second[0]
                
                # If both are difficult emotions, prioritize more visible one
                difficult_pairs = [('fear', 'surprise'), ('anger', 'disgust'), ('sad', 'disgust')]
                for pair in difficult_pairs:
                    if first[0] in pair and second[0] in pair:
                        if first[0] == 'surprise' or first[0] == 'anger' or first[0] == 'sad':
                            return first[0]
                        else:
                            return second[0]
        
        # Return original emotion if no adjustment was applied
        return emotion
    
    except Exception as e:
        print(f"Error in emotion improvement: {e}")
        return emotion  # Return original emotion if there was an error 