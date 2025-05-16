import cv2
import numpy as np
import os
from emotion_detection import get_emotion_vector, detect_emotion
from gender_detection import get_gender_vector, detect_gender
from skin_classification import get_skin_vector, classify_skin_color

# Load the face cascade classifier
face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(face_cascade_path)

def extract_face(image):
    """
    Extract face from an image using OpenCV
    
    Args:
        image: Input image
        
    Returns:
        numpy.ndarray: Extracted face region or None if no face found
    """
    # Input validation
    if image is None or image.size == 0:
        print("Invalid image in face extraction")
        return None
    
    try:
        # Convert to grayscale for face detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Improve contrast for better face detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        
        # Detect faces with different scales
        faces = []
        for scale in [1.1, 1.2, 1.3]:
            detected = face_cascade.detectMultiScale(
                enhanced_gray,
                scaleFactor=scale,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(detected) > 0:
                faces = detected
                break
        
        # If no face detected, try with less strict parameters
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(
                enhanced_gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(20, 20)
            )
        
        # If still no face, return center region of image
        if len(faces) == 0:
            h, w = image.shape[:2]
            center_x, center_y = w // 2, h // 2
            face_w, face_h = w // 2, h // 2
            x1 = max(0, center_x - face_w // 2)
            y1 = max(0, center_y - face_h // 2)
            x2 = min(w, x1 + face_w)
            y2 = min(h, y1 + face_h)
            
            print("No face detected, using center region of image")
            if len(image.shape) == 3:
                return image[y1:y2, x1:x2]
            else:
                return cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_GRAY2BGR)
        
        # Get the largest face (by area)
        if len(faces) > 1:
            print(f"Multiple faces detected ({len(faces)}), using the largest one")
            largest_area = 0
            largest_face = None
            for (x, y, w, h) in faces:
                if w * h > largest_area:
                    largest_area = w * h
                    largest_face = (x, y, w, h)
            faces = [largest_face]
        
        # Extract the face region
        (x, y, w, h) = faces[0]
        
        # Add margin to include whole face
        margin_percent = 0.3
        x_margin = int(w * margin_percent)
        y_margin = int(h * margin_percent)
        
        x1 = max(0, x - x_margin)
        y1 = max(0, y - y_margin)
        x2 = min(image.shape[1], x + w + x_margin)
        y2 = min(image.shape[0], y + h + y_margin)
        
        # Extract the face with margin
        if len(image.shape) == 3:
            face_region = image[y1:y2, x1:x2]
        else:
            face_region = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_GRAY2BGR)
        
        # Resize to standard size
        face_region = cv2.resize(face_region, (128, 128))
        
        return face_region
        
    except Exception as e:
        print(f"Error in face extraction: {e}")
        return None

def create_face_encoding(face_image):
    """
    Create a 128-dimensional vector encoding of the face using OpenCV
    
    Args:
        face_image: Input face image
        
    Returns:
        numpy.ndarray: 128-dimensional face encoding vector
    """
    # Input validation
    if face_image is None or face_image.size == 0:
        print("Invalid face image for encoding")
        # Trả về vector với các giá trị ngẫu nhiên nhỏ thay vì toàn 0
        return np.random.normal(0, 0.1, 128)
    
    try:
        # Ensure image is in RGB format
        if len(face_image.shape) < 3:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
        
        # Resize to standard size
        face_image = cv2.resize(face_image, (128, 128))
        
        # Convert to grayscale
        gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Tăng cường độ tương phản để có feature tốt hơn
        gray_face = cv2.equalizeHist(gray_face)
        
        # Sử dụng phương pháp đơn giản hơn để tránh lỗi broadcast
        # Chia ảnh thành 16 vùng, tính trung bình và độ lệch chuẩn cho mỗi vùng
        regions_h = 4
        regions_w = 4
        region_h = gray_face.shape[0] // regions_h
        region_w = gray_face.shape[1] // regions_w
        
        # Khởi tạo encoding
        encoding = np.zeros(128)
        idx = 0
        
        # Tạo đặc trưng từ các vùng khác nhau của khuôn mặt
        for i in range(regions_h):
            for j in range(regions_w):
                # Lấy vùng
                y_start = i * region_h
                y_end = min((i + 1) * region_h, gray_face.shape[0])
                x_start = j * region_w
                x_end = min((j + 1) * region_w, gray_face.shape[1])
                
                region = gray_face[y_start:y_end, x_start:x_end]
                
                # Đảm bảo vùng không trống
                if region.size == 0:
                    # Nếu vùng trống, điền giá trị ngẫu nhiên
                    features = np.random.normal(0, 0.1, 8)
                else:
                    # Tính toán gradient
                    try:
                        gx = cv2.Sobel(region, cv2.CV_32F, 1, 0)
                        gy = cv2.Sobel(region, cv2.CV_32F, 0, 1)
                        
                        # Tính độ lớn và hướng gradient
                        magnitude = cv2.magnitude(gx, gy)
                        angle = cv2.phase(gx, gy, angleInDegrees=True)
                        
                        # Tạo histogram theo 8 hướng (0-45, 45-90, ...)
                        hist = np.zeros(8)
                        
                        # Kiểm tra magnitude và angle có cùng kích thước
                        if magnitude.shape == angle.shape and magnitude.size > 0:
                            for a, m in zip(angle.flatten(), magnitude.flatten()):
                                bin_idx = int(a // 45) % 8
                                hist[bin_idx] += m
                            
                            # Chuẩn hóa histogram
                            hist_sum = np.sum(hist)
                            if hist_sum > 0:
                                hist = hist / hist_sum
                        else:
                            # Nếu kích thước không khớp, tạo histogram ngẫu nhiên
                            hist = np.random.normal(0, 0.1, 8)
                        
                        features = hist
                    except Exception as e:
                        print(f"Error processing region {i},{j}: {e}")
                        # Sử dụng giá trị ngẫu nhiên nếu xử lý vùng gặp lỗi
                        features = np.random.normal(0, 0.1, 8)
                
                # Thêm vào encoding nếu chỉ số hợp lệ
                if idx + 8 <= 128:
                    encoding[idx:idx+8] = features
                else:
                    print(f"Warning: Index {idx} out of bounds for encoding")
                
                idx += 8
        
        # Đảm bảo không còn chỉ số nào vượt quá kích thước của encoding
        if idx < 128:
            # Điền phần còn lại bằng giá trị ngẫu nhiên
            encoding[idx:] = np.random.normal(0, 0.1, 128 - idx)
        
        # Chuẩn hóa toàn bộ encoding
        norm = np.linalg.norm(encoding)
        if norm > 0:
            encoding = encoding / norm
        else:
            # Nếu norm bằng 0, tạo một vector ngẫu nhiên đã chuẩn hóa
            random_encoding = np.random.normal(0, 1, 128)
            random_norm = np.linalg.norm(random_encoding)
            encoding = random_encoding / random_norm
        
        # Kiểm tra NaN và thay thế bằng 0
        if np.isnan(encoding).any():
            print("Warning: NaN values in face encoding, replacing with zeros")
            encoding = np.nan_to_num(encoding)
        
        return encoding
        
    except Exception as e:
        print(f"Error creating face encoding: {e}")
        # Trả về vector ngẫu nhiên đã chuẩn hóa thay vì toàn 0
        random_encoding = np.random.normal(0, 1, 128)
        return random_encoding / np.linalg.norm(random_encoding)

def extract_features(image_path=None, image_array=None):
    """
    Extract all features from an image and create a combined 176-dimensional vector
    
    Args:
        image_path: Path to image file (optional if image_array is provided)
        image_array: Image as numpy array (optional if image_path is provided)
        
    Returns:
        dict: Dictionary containing all extracted features and vectors
    """
    result = {
        'face_found': False,
        'gender': 'Unknown',
        'gender_confidence': 0.0,
        'skin_color': 'Unknown',
        'skin_confidence': 0.0,
        'emotion': 'Unknown',
        'emotion_confidence': 0.0,
        'face_encoding': None,
        'gender_vector': None,
        'skin_vector': None,
        'emotion_vector': None,
        'combined_vector': None
    }
    
    try:
        # Load image
        if image_path is not None:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image from path: {image_path}")
                return result
        elif image_array is not None:
            image = image_array.copy()
        else:
            print("No image provided")
            return result
        
        # Validate image
        if image is None or image.size == 0:
            print("Invalid image")
            return result
        
        # Extract face
        face_image = extract_face(image)
        if face_image is None:
            print("No face found in image")
            return result
        
        # Mark as face found
        result['face_found'] = True
        
        # Extract face encoding (128-dimensional vector)
        face_encoding = create_face_encoding(face_image)
        result['face_encoding'] = face_encoding
        
        # Detect gender and create gender vector (16-dimensional) 
        # Sử dụng face_encoding để cải thiện độ chính xác của phát hiện giới tính
        gender, gender_confidence = detect_gender(face_image, face_encoding=face_encoding)
        result['gender'] = gender
        result['gender_confidence'] = gender_confidence
        gender_vector = get_gender_vector(face_image, face_encoding=face_encoding)
        result['gender_vector'] = gender_vector
        
        # Detect skin color and create skin vector (16-dimensional)
        skin_color, skin_confidence = classify_skin_color(face_image)
        result['skin_color'] = skin_color
        result['skin_confidence'] = skin_confidence
        skin_vector = get_skin_vector(face_image)
        result['skin_vector'] = skin_vector
        
        # Detect emotion and create emotion vector (16-dimensional)
        emotion, emotion_confidence = detect_emotion(face_image)
        result['emotion'] = emotion
        result['emotion_confidence'] = emotion_confidence
        emotion_vector = get_emotion_vector(face_image)
        result['emotion_vector'] = emotion_vector
        
        # Create combined vector (176-dimensional)
        # Concatenate face_encoding (128), gender_vector (16), skin_vector (16), emotion_vector (16)
        combined_vector = np.concatenate([face_encoding, gender_vector, skin_vector, emotion_vector])
        
        # Ensure the vector is exactly 176 dimensions
        if combined_vector.shape[0] != 176:
            print(f"Warning: Combined vector has {combined_vector.shape[0]} dimensions, expected 176")
            # Padding if necessary
            if combined_vector.shape[0] < 176:
                padding = np.zeros(176 - combined_vector.shape[0])
                combined_vector = np.concatenate([combined_vector, padding])
            # Truncating if necessary
            elif combined_vector.shape[0] > 176:
                combined_vector = combined_vector[:176]
        
        # Check for NaN values and replace with zeros
        if np.isnan(combined_vector).any():
            print("Warning: NaN values detected in combined vector, replacing with zeros")
            combined_vector = np.nan_to_num(combined_vector)
        
        result['combined_vector'] = combined_vector
        
        print(f"Features extracted successfully: {gender} {skin_color} {emotion}")
        return result
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return result