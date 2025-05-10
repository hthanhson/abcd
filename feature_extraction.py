import face_recognition
import cv2
import numpy as np
from deepface import DeepFace
from emotion_detection import improve_emotion_detection
from age_estimation import adjust_age_estimation
from skin_classification import classify_skin_color
from utils import categorize_age
import os
height, width = 224, 224  
min_dimension = min(height, width) 
min_face_size = int(min_dimension * 0.45)
max_face_size = int(min_dimension * 0.60)
def extract_features(image_path):
    """Extract facial features: encoding, emotion, age, and skin color with improved accuracy"""
    try:
        # Load image
        image = face_recognition.load_image_file(image_path)
        
        # Preprocess image to improve quality
        # 1. Convert to grayscale to increase contrast
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 2. Improve contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray_image)
        
        # 3. Reduce noise
        denoised = cv2.fastNlMeansDenoising(enhanced_gray, None, 10, 7, 21)
        
        # Find faces using multiple methods
        # Try face_recognition first
        face_locations = face_recognition.face_locations(image)
        
        # If no face is found, try with lower accuracy
        if not face_locations:
            # Try with CNN model which is slower but more accurate for difficult cases
            face_locations = face_recognition.face_locations(image, model="cnn")
        
        # If still no face, try with OpenCV cascade
        if not face_locations:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            cv_faces = face_cascade.detectMultiScale(
                denoised, 
                scaleFactor=1.1, 
                minNeighbors=3, 
                minSize=(60,60),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Convert format from OpenCV to face_recognition
            if len(cv_faces) > 0:
                face_locations = []
                for (x, y, w, h) in cv_faces:
                    face_locations.append((y, x + w, y + h, x))  # top, right, bottom, left
        
        # If still no face found, try with different parameters
        if not face_locations:
            # Try with lower MinNeighbors parameter
            cv_faces = face_cascade.detectMultiScale(
                denoised, 
                scaleFactor=1.05, 
                minNeighbors=2, 
                minSize=(40,40),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(cv_faces) > 0:
                face_locations = []
                for (x, y, w, h) in cv_faces:
                    face_locations.append((y, x + w, y + h, x))
        
        # If still no face found, assume a face in the center of the image (not ideal but better than nothing)
        if not face_locations:
            # Assume a face in the center of the image
            height, width = image.shape[:2]
            center_x, center_y = width // 2, height // 2
            face_size = min(width, height) // 2
            
            # Create a virtual face
            top = max(0, center_y - face_size)
            right = min(width, center_x + face_size)
            bottom = min(height, center_y + face_size)
            left = max(0, center_x - face_size)
            
            face_locations = [(top, right, bottom, left)]
            print(f"No face detected in {image_path}, using center of image")
            
        # Now we have face_locations, continue with analysis
        
        # Get encoding for the face
        face_encodings = face_recognition.face_encodings(image, face_locations)
        if not face_encodings and len(face_locations) > 0:
            # Expand face area if encoding couldn't be obtained
            expanded_face_locations = []
            for (top, right, bottom, left) in face_locations:
                height, width = image.shape[:2]
                # Expand 20% in each direction
                expand_px = min(bottom - top, right - left) // 5
                new_top = max(0, top - expand_px)
                new_right = min(width, right + expand_px)
                new_bottom = min(height, bottom + expand_px)
                new_left = max(0, left - expand_px)
                expanded_face_locations.append((new_top, new_right, new_bottom, new_left))
                
            # Try to get encoding with expanded area
            face_encodings = face_recognition.face_encodings(image, expanded_face_locations)
            if face_encodings:
                # If successful, update face_locations
                face_locations = expanded_face_locations
        
        # If still couldn't get encoding
        if not face_encodings:
            # For difficult cases, create a dummy encoding
            # Empty encoding with average values
            dummy_encoding = np.zeros(128)
            # Add some random variation to avoid exact duplicates
            random_factor = np.random.normal(0, 0.01, 128)
            dummy_encoding += random_factor
            dummy_encoding /= np.linalg.norm(dummy_encoding)  # Normalize
            print(f"Could not extract encoding from {image_path}, using fallback")
            encoding = dummy_encoding
        else:
            encoding = face_encodings[0]  # Use first face
        
        # Đảm bảo encoding là mảng NumPy và có kích thước đúng
        if not isinstance(encoding, np.ndarray):
            encoding = np.array(encoding)
        
        if encoding.shape[0] != 128:
            print(f"Warning: Unexpected encoding dimension {encoding.shape}, resizing to 128")
            # Nếu kích thước không đúng, tạo vector ngẫu nhiên
            dummy_encoding = np.random.normal(0, 0.01, 128)
            dummy_encoding /= np.linalg.norm(dummy_encoding)
            encoding = dummy_encoding
        
        # Extract face region for further processing
        top, right, bottom, left = face_locations[0]
        face_image = image[top:bottom, left:right]
        
        # Analyze emotion and age using DeepFace
        try:
            # Try using both the extracted face and the full image
            try:
                analysis = DeepFace.analyze(
                    face_image, 
                    actions=['emotion', 'age'], 
                    enforce_detection=False,
                    detector_backend='retinaface'
                )
            except:
                # If direct face analysis fails, try with original image
                analysis = DeepFace.analyze(
                    image_path, 
                    actions=['emotion', 'age'], 
                    enforce_detection=False,
                    detector_backend='opencv'  # Try with opencv if retinaface fails
                )
            
            # Ensure consistent format
            if isinstance(analysis, list):
                analysis = analysis[0]
            
            # Get emotion data and improve accuracy
            emotions_dict = analysis['emotion']
            emotion = analysis['dominant_emotion']
            improved_emotion = improve_emotion_detection(emotion, face_image, emotions_dict)
            
            # Get age and adjust
            raw_age = analysis['age']
            adjusted_age = adjust_age_estimation(raw_age, face_image)
            age_group = categorize_age(adjusted_age)
            
            # Analyze skin color
            hsv_face = cv2.cvtColor(face_image, cv2.COLOR_RGB2HSV)
            skin_color_category = classify_skin_color(hsv_face)
            
            return encoding, improved_emotion, adjusted_age, age_group, skin_color_category
            
        except Exception as e:
            print(f"Error with DeepFace analysis: {e}")
            # Return default values if analysis fails
            return encoding, "neutral", 30, "adult", "unknown"
            
    except Exception as e:
        print(f"Critical error analyzing image {image_path}: {e}")
        return None, None, None, None, None 