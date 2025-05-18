import cv2
import numpy as np
import hashlib

# Bộ nhớ đệm toàn cục để lưu trữ kết quả dựa trên hash của ảnh
_emotion_cache = {}

def get_image_hash(image):
    """
    Tạo hash duy nhất cho ảnh để lưu vào bộ nhớ đệm
    """
    if image is None or image.size == 0:
        return "empty_image"
    
    # Resize nhỏ lại để tính hash nhanh hơn và ít bị ảnh hưởng bởi nhiễu
    small_img = cv2.resize(image, (32, 32))
    # Chuyển đổi sang grayscale nếu cần
    if len(small_img.shape) == 3:
        small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    
    # Tính toán hash của ảnh đã được chuẩn hóa
    img_bytes = small_img.tobytes()
    img_hash = hashlib.md5(img_bytes).hexdigest()
    return img_hash

def detect_emotion(face_image, force_recalculate=False):
    """
    Detect emotion from a face image - optimized for 224x224 input images
    with improved mouth shape detection and reduced neutral bias
    
    Args:
        face_image: Input face image
        force_recalculate: Force recalculation even if image is in cache
    
    Returns:
        tuple: (emotion, confidence) - detected emotion and confidence level
    """
    # Input validation
    if face_image is None or face_image.size == 0:
        print("Invalid image in emotion detection, using default (Neutral)")
        return "Neutral", 0.7
    
    # Kiểm tra bộ nhớ đệm nếu không bị bắt buộc tính toán lại
    if not force_recalculate:
        img_hash = get_image_hash(face_image)
        if img_hash in _emotion_cache:
            cached_result = _emotion_cache[img_hash]
            print(f"Using cached emotion result: {cached_result[0]} (confidence: {cached_result[1]:.2f})")
            return cached_result
    
    try:
        # Ensure image is in grayscale format for emotion detection
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image.copy()
        
        # Save original dimensions
        original_h, original_w = gray.shape
        
        # Chuẩn hóa kích thước ảnh
        resized = cv2.resize(gray, (224, 224))
        
        # Chuẩn hóa độ sáng và tương phản
        # 1. Giảm nhiễu với Gaussian blur nhẹ
        blurred = cv2.GaussianBlur(resized, (3, 3), 0)
        
        # 2. Cân bằng histogram thích ứng
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # 3. Chuẩn hóa về khoảng sáng-tối nhất định
        face = cv2.normalize(enhanced, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        
        # Extract facial features relevant to emotion with improved regions for 224x224
        h, w = face.shape
        
        # Eye regions (improved positions for 224x224)
        left_eye_region = face[int(h*0.22):int(h*0.43), int(w*0.15):int(w*0.45)]
        right_eye_region = face[int(h*0.22):int(h*0.43), int(w*0.55):int(w*0.85)]
        
        # Mouth region - more precise for 224x224 - MỞ RỘNG vùng miệng để phát hiện tốt hơn
        mouth_region = face[int(h*0.60):int(h*0.90), int(w*0.25):int(w*0.75)]
        
        # Eyebrow regions - better defined for 224x224
        left_eyebrow_region = face[int(h*0.12):int(h*0.22), int(w*0.15):int(w*0.45)]
        right_eyebrow_region = face[int(h*0.12):int(h*0.22), int(w*0.55):int(w*0.85)]
        
        # Nose region - better defined for 224x224
        nose_region = face[int(h*0.35):int(h*0.62), int(w*0.35):int(w*0.65)]
        
        # Forehead region - new for 224x224
        forehead_region = face[int(h*0.05):int(h*0.15), int(w*0.25):int(w*0.75)]
        
        # Calculate features
        # 1. Standard deviation of pixel values in regions (texture/expression intensity)
        mouth_std = np.std(mouth_region) if mouth_region.size > 0 else 0
        left_eye_std = np.std(left_eye_region) if left_eye_region.size > 0 else 0
        right_eye_std = np.std(right_eye_region) if right_eye_region.size > 0 else 0
        left_eyebrow_std = np.std(left_eyebrow_region) if left_eyebrow_region.size > 0 else 0
        right_eyebrow_std = np.std(right_eyebrow_region) if right_eyebrow_region.size > 0 else 0
        
        # 2. Edge density in regions (for expression lines) - improved for 224x224
        mouth_edges = cv2.Sobel(mouth_region, cv2.CV_64F, 1, 1) if mouth_region.size > 0 else np.zeros((1, 1))
        mouth_edge_mean = np.mean(np.abs(mouth_edges))
        
        # 3. Gradient magnitude for eyebrows and mouth corners (expression indicators)
        face_gradient_x = cv2.Sobel(face, cv2.CV_64F, 1, 0)
        face_gradient_y = cv2.Sobel(face, cv2.CV_64F, 0, 1)
        gradient_magnitude = cv2.magnitude(face_gradient_x, face_gradient_y)
        
        # 4. Mouth openness (ratio of bright to dark pixels, threshold-based) - CẢI TIẾN phát hiện khẩu hình
        # Dùng nhiều ngưỡng khác nhau để phát hiện trạng thái miệng tốt hơn
        _, mouth_thresh1 = cv2.threshold(mouth_region, 100, 255, cv2.THRESH_BINARY) if mouth_region.size > 0 else (None, np.zeros((1, 1)))
        _, mouth_thresh2 = cv2.threshold(mouth_region, 150, 255, cv2.THRESH_BINARY) if mouth_region.size > 0 else (None, np.zeros((1, 1)))
        
        mouth_open_ratio1 = np.sum(mouth_thresh1) / (mouth_region.size * 255) if mouth_region.size > 0 else 0
        mouth_open_ratio2 = np.sum(mouth_thresh2) / (mouth_region.size * 255) if mouth_region.size > 0 else 0
        
        # Kết hợp các ngưỡng để có đánh giá chính xác hơn
        mouth_open_ratio = (mouth_open_ratio1 + mouth_open_ratio2) / 2
        
        # 5. CẢI TIẾN: Phát hiện đường cong miệng (cười/buồn)
        if mouth_region.size > 0:
            # Chia miệng thành nửa trên và nửa dưới
            h_mouth, w_mouth = mouth_region.shape
            upper_lip = mouth_region[:int(h_mouth/2), :]
            lower_lip = mouth_region[int(h_mouth/2):, :]
            
            # Tính độ cong của môi (gradient dọc)
            upper_lip_gradient = cv2.Sobel(upper_lip, cv2.CV_64F, 0, 1) if upper_lip.size > 0 else np.zeros((1, 1))
            lower_lip_gradient = cv2.Sobel(lower_lip, cv2.CV_64F, 0, 1) if lower_lip.size > 0 else np.zeros((1, 1))
            
            # Tính giá trị trung bình của gradients
            upper_lip_curve = np.mean(upper_lip_gradient) if upper_lip.size > 0 else 0
            lower_lip_curve = np.mean(lower_lip_gradient) if lower_lip.size > 0 else 0
            
            # Độ cong tổng thể (+ = cười, - = buồn)
            mouth_curve = upper_lip_curve - lower_lip_curve
        else:
            mouth_curve = 0
        
        # 6. CẢI TIẾN: Phát hiện góc miệng (quan trọng cho nụ cười và buồn bã)
        if mouth_region.size > 0:
            h_mouth, w_mouth = mouth_region.shape
            
            # Lấy góc miệng trái và phải
            left_corner = mouth_region[int(h_mouth*0.5):int(h_mouth*0.8), :int(w_mouth*0.3)]
            right_corner = mouth_region[int(h_mouth*0.5):int(h_mouth*0.8), int(w_mouth*0.7):]
            
            # Tính gradient tại các góc
            left_corner_grad = cv2.Sobel(left_corner, cv2.CV_64F, 1, 1) if left_corner.size > 0 else np.zeros((1, 1))
            right_corner_grad = cv2.Sobel(right_corner, cv2.CV_64F, 1, 1) if right_corner.size > 0 else np.zeros((1, 1))
            
            # Giá trị trung bình
            left_corner_value = np.mean(np.abs(left_corner_grad)) if left_corner.size > 0 else 0
            right_corner_value = np.mean(np.abs(right_corner_grad)) if right_corner.size > 0 else 0
            
            # Cường độ tại các góc miệng
            mouth_corner_intensity = (left_corner_value + right_corner_value) / 2
        else:
            mouth_corner_intensity = 0
            
        # 7. Eye openness (new feature for 224x224)
        _, left_eye_thresh = cv2.threshold(left_eye_region, 100, 255, cv2.THRESH_BINARY_INV) if left_eye_region.size > 0 else (None, np.zeros((1, 1)))
        _, right_eye_thresh = cv2.threshold(right_eye_region, 100, 255, cv2.THRESH_BINARY_INV) if right_eye_region.size > 0 else (None, np.zeros((1, 1)))
        left_eye_open_ratio = np.sum(left_eye_thresh) / (left_eye_region.size * 255) if left_eye_region.size > 0 else 0
        right_eye_open_ratio = np.sum(right_eye_thresh) / (right_eye_region.size * 255) if right_eye_region.size > 0 else 0
        eye_open_ratio = (left_eye_open_ratio + right_eye_open_ratio) / 2
        
        # 8. Eyebrow positioning (new feature for 224x224)
        left_eyebrow_mean = np.mean(left_eyebrow_region) if left_eyebrow_region.size > 0 else 0
        right_eyebrow_mean = np.mean(right_eyebrow_region) if right_eyebrow_region.size > 0 else 0
        eyebrow_mean = (left_eyebrow_mean + right_eyebrow_mean) / 2
        
        # 9. Forehead wrinkling (new feature for 224x224)
        forehead_edges = cv2.Sobel(forehead_region, cv2.CV_64F, 1, 1) if forehead_region.size > 0 else np.zeros((1, 1))
        forehead_edge_mean = np.mean(np.abs(forehead_edges))
        
        # Print debug information
        print(f"Emotion features (224x224 optimized):")
        print(f"- Mouth std: {mouth_std:.2f}")
        print(f"- Eye std (L/R): {left_eye_std:.2f}/{right_eye_std:.2f}")
        print(f"- Mouth edge mean: {mouth_edge_mean:.2f}")
        print(f"- Mouth open ratio: {mouth_open_ratio:.2f}")
        print(f"- Mouth curve: {mouth_curve:.2f}")
        print(f"- Mouth corner intensity: {mouth_corner_intensity:.2f}")
        print(f"- Eye open ratio: {eye_open_ratio:.2f}")
        print(f"- Eyebrow mean: {eyebrow_mean:.2f}")
        print(f"- Forehead edge mean: {forehead_edge_mean:.2f}")
        
        # Calculate base scores for each emotion based on extracted features
        # Higher values indicate stronger presence of the emotion
        # GIẢM độ ưu tiên của trạng thái neutral từ 0.5 xuống 0.3
        happy_score = 0.0
        sad_score = 0.0
        angry_score = 0.0
        surprised_score = 0.0
        fearful_score = 0.0
        disgusted_score = 0.0
        neutral_score = 0.3  # Giảm baseline cho neutral từ 0.5 xuống 0.3
        
        # Happy indicators: Open mouth, raised cheeks, specific eye shape - CƯỜNG HÓA tín hiệu vui vẻ
        # Sử dụng mouth_curve làm chỉ báo chính, giá trị dương = cười
        if mouth_curve > 1.5:  # Miệng cong lên = cười
            happy_score += 0.5
            neutral_score -= 0.2
            print("- Happy indicators detected: smile curve (strong)")
        elif mouth_curve > 0.8:  # Miệng cong lên nhẹ
            happy_score += 0.3
            neutral_score -= 0.1
            print("- Happy indicators detected: smile curve (mild)")
            
        # Bổ sung thêm phát hiện nụ cười dựa trên các chỉ số khác
        if mouth_open_ratio > 0.3 and mouth_corner_intensity > 40:
            happy_score += 0.3
            neutral_score -= 0.1
            print("- Happy indicators detected: open mouth with strong corners")
        
        # Sad indicators: Downturned mouth corners, droopy eyes - CƯỜNG HÓA tín hiệu buồn bã
        # Sử dụng mouth_curve làm chỉ báo chính, giá trị âm = buồn
        if mouth_curve < -1.5:  # Miệng cong xuống rõ rệt
            sad_score += 0.5
            happy_score -= 0.2
            neutral_score -= 0.2
            print("- Sad indicators detected: downturned mouth (strong)")
        elif mouth_curve < -0.8:  # Miệng cong xuống nhẹ
            sad_score += 0.3
            happy_score -= 0.1
            print("- Sad indicators detected: downturned mouth (mild)")
        
        # Bổ sung phát hiện buồn bã từ các đặc trưng khác
        if mouth_edge_mean > 30 and mouth_open_ratio < 0.25 and eyebrow_mean < 100:
            sad_score += 0.2
            print("- Sad indicators detected: low eyebrows and closed mouth")
        
        # Angry indicators: Furrowed brow, tight mouth - optimized for 224x224
        eyebrow_edges = cv2.Sobel(np.vstack([left_eyebrow_region, right_eyebrow_region]), cv2.CV_64F, 1, 1) if left_eyebrow_region.size > 0 and right_eyebrow_region.size > 0 else np.zeros((1, 1))
        eyebrow_edge_mean = np.mean(np.abs(eyebrow_edges))
        
        # GIẢM ngưỡng cho angry để phát hiện tốt hơn
        if eyebrow_edge_mean > 45 and mouth_edge_mean > 30 and mouth_open_ratio < 0.3:
            angry_score += 0.45
            neutral_score -= 0.2
            print(f"- Angry indicators detected: eyebrow furrow ({eyebrow_edge_mean:.2f}) and tight mouth")
        
        # Surprised indicators: Wide eyes, open mouth - optimized for 224x224
        # GIẢM ngưỡng để phát hiện ngạc nhiên tốt hơn
        if eye_open_ratio > 0.4 and mouth_open_ratio > 0.35:
            surprised_score += 0.55
            neutral_score -= 0.3
            print("- Surprised indicators detected: wide eyes and open mouth")
        
        # Fearful indicators: Wide eyes, tense mouth - optimized for 224x224
        if eye_open_ratio > 0.35 and mouth_edge_mean > 30 and forehead_edge_mean > 25:
            fearful_score += 0.4
            neutral_score -= 0.1
            print("- Fearful indicators detected: wide eyes, tense mouth, and forehead wrinkles")
        
        # Disgusted indicators: Wrinkled nose, raised upper lip - optimized for 224x224
        nose_edges = cv2.Sobel(nose_region, cv2.CV_64F, 1, 1) if nose_region.size > 0 else np.zeros((1, 1))
        nose_edge_mean = np.mean(np.abs(nose_edges))
        
        if nose_edge_mean > 50 and mouth_edge_mean > 35:
            disgusted_score += 0.4
            neutral_score -= 0.1
            print(f"- Disgusted indicators detected: nose wrinkles ({nose_edge_mean:.2f}) and grimace")
        
        # Neutral indicators: Balanced features, moderate values - improved for 224x224
        # THÊM YÊU CẦU chặt chẽ hơn để xác định neutral
        if (abs(left_eye_std - right_eye_std) < 10 and 
            abs(mouth_curve) < 0.5 and   # Miệng không cong nhiều
            mouth_edge_mean < 25 and 
            eyebrow_edge_mean < 35 and
            forehead_edge_mean < 20 and
            mouth_open_ratio < 0.3):      # Miệng không mở quá to
            
            neutral_score += 0.3
            print("- Neutral indicators detected: balanced features and low intensity")
        
        # Combine the features into emotion scores
        emotion_scores = {
            "Happy": happy_score,
            "Sad": sad_score,
            "Angry": angry_score,
            "Surprised": surprised_score,
            "Fearful": fearful_score,
            "Disgusted": disgusted_score,
            "Neutral": neutral_score
        }
        
        # THÊM ĐK: Nếu không cảm xúc nào vượt trội, tìm cảm xúc gần nhất với miệng hiện tại
        max_score = max(emotion_scores.values())
        if max_score < 0.3:  # Nếu điểm cao nhất quá thấp
            # Xác định cảm xúc dựa vào hình dạng miệng
            if mouth_curve > 0:
                emotion_scores["Happy"] += 0.2
                print("- Low confidence, boosting Happy based on mouth shape")
            elif mouth_curve < 0:
                emotion_scores["Sad"] += 0.2
                print("- Low confidence, boosting Sad based on mouth shape")
            elif mouth_open_ratio > 0.3:
                emotion_scores["Surprised"] += 0.2
                print("- Low confidence, boosting Surprised based on mouth openness")
        
        # Find emotion with highest score
        emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[emotion]
        
        # Ensure minimum confidence and apply ceiling
        if confidence < 0.25:
            # ĐÃ THAY ĐỔI: Không luôn mặc định về Neutral mà tìm cảm xúc tốt nhất từ khẩu hình
            if mouth_curve > 1.0:
                emotion = "Happy"
                confidence = 0.6
                print(f"- Low confidence, using mouth curve to determine: {emotion}")
            elif mouth_curve < -0.8:
                emotion = "Sad"
                confidence = 0.6
                print(f"- Low confidence, using mouth curve to determine: {emotion}")
            elif mouth_open_ratio > 0.35:
                emotion = "Surprised"
                confidence = 0.6
                print(f"- Low confidence, using mouth openness to determine: {emotion}")
            else:
                emotion = "Neutral"
                confidence = 0.6
                print(f"- Low confidence, defaulting to: {emotion}")
        else:
            # Cap confidence at 0.95
            confidence = min(confidence, 0.95)
        
        print(f"Detected emotion: {emotion} (confidence: {confidence:.2f})")
        
        # Lưu kết quả vào bộ nhớ đệm
        img_hash = get_image_hash(face_image)
        _emotion_cache[img_hash] = (emotion, confidence)
        
        return emotion, confidence
        
    except Exception as e:
        print(f"Error in emotion detection: {e}")
        return "Neutral", 0.7

def get_emotion_vector(face_image, vector_length=16):
    """
    Create feature vector for emotion from face image - optimized for 224x224 images
    
    Args:
        face_image: Face image
        vector_length: Length of output vector
        
    Returns:
        ndarray: Emotion feature vector (16-dimensional)
    """
    # Initialize feature vector
    emotion_vector = np.zeros(vector_length, dtype=float)
    
    # Validate input
    if face_image is None or face_image.size == 0:
        return emotion_vector
    
    try:
        # Detect emotion and confidence - sử dụng bộ nhớ đệm
        emotion, confidence = detect_emotion(face_image)
        
        # Define emotion indices for one-hot encoding
        emotion_mapping = {
            "Happy": 0,
            "Sad": 1,
            "Angry": 2,
            "Surprised": 3,
            "Fearful": 4,
            "Disgusted": 5,
            "Neutral": 6,
            "Unknown": 7
        }
        
        # Get index of detected emotion
        emotion_index = emotion_mapping.get(emotion, 7)  # Default to unknown
        
        # Create a copy for analysis
        if len(face_image.shape) == 3:
            face_copy = face_image.copy()
            gray = cv2.cvtColor(face_copy, cv2.COLOR_BGR2GRAY)
        else:
            face_copy = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
            gray = face_image.copy()
        
        # Resize for improved feature extraction at 224x224
        gray = cv2.resize(gray, (224, 224))
        
        # Apply CLAHE for better contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        
        # Get facial regions - optimized for 224x224
        h, w = gray.shape
        eye_region = gray[int(h*0.22):int(h*0.43), :]
        mouth_region = gray[int(h*0.60):int(h*0.85), int(w*0.28):int(w*0.72)]
        eyebrow_region = gray[int(h*0.12):int(h*0.22), :]
        nose_region = gray[int(h*0.35):int(h*0.62), int(w*0.35):int(w*0.65)]
        
        # Extract features - enhanced for 224x224
        features = []
        
        # 1. Eye region statistics
        if eye_region.size > 0:
            eye_mean = np.mean(eye_region) / 255.0
            eye_std = np.std(eye_region) / 100.0
            features.extend([float(eye_mean), float(eye_std)])
        else:
            features.extend([0.5, 0.1])
        
        # 2. Mouth region statistics + KHẨU HÌNH
        if mouth_region.size > 0:
            mouth_mean = np.mean(mouth_region) / 255.0
            mouth_std = np.std(mouth_region) / 100.0
            
            # Thêm đặc trưng khẩu hình
            h_mouth, w_mouth = mouth_region.shape
            upper_lip = mouth_region[:int(h_mouth/2), :]
            lower_lip = mouth_region[int(h_mouth/2):, :]
            
            upper_lip_gradient = cv2.Sobel(upper_lip, cv2.CV_64F, 0, 1) if upper_lip.size > 0 else np.zeros((1, 1))
            lower_lip_gradient = cv2.Sobel(lower_lip, cv2.CV_64F, 0, 1) if lower_lip.size > 0 else np.zeros((1, 1))
            
            upper_lip_curve = float(np.mean(upper_lip_gradient) / 50.0) if upper_lip.size > 0 else 0
            lower_lip_curve = float(np.mean(lower_lip_gradient) / 50.0) if lower_lip.size > 0 else 0
            
            # Thêm vào vector đặc trưng
            features.extend([float(mouth_mean), float(mouth_std), float(upper_lip_curve), float(lower_lip_curve)])
        else:
            features.extend([0.5, 0.1, 0.0, 0.0])
        
        # 3. Gradient information - improved for 224x224
        gradient_x = cv2.Sobel(equalized, cv2.CV_64F, 1, 0)
        gradient_y = cv2.Sobel(equalized, cv2.CV_64F, 0, 1)
        
        gradient_mag = cv2.magnitude(gradient_x, gradient_y)
        gradient_mean = float(np.mean(gradient_mag) / 150.0)  # Adjusted normalization for 224x224
        gradient_std = float(np.std(gradient_mag) / 150.0)    # Adjusted normalization for 224x224
        features.extend([gradient_mean, gradient_std])
        
        # 4. New feature: Eyebrow intensity (indicative of many emotions)
        if eyebrow_region.size > 0:
            eyebrow_gradient_x = cv2.Sobel(eyebrow_region, cv2.CV_64F, 1, 0)
            eyebrow_gradient_y = cv2.Sobel(eyebrow_region, cv2.CV_64F, 0, 1)
            eyebrow_grad_mag = cv2.magnitude(eyebrow_gradient_x, eyebrow_gradient_y)
            eyebrow_intensity = float(np.mean(eyebrow_grad_mag) / 150.0)
            features.append(eyebrow_intensity)
        else:
            features.append(0.1)
            
        # 5. Confidence of emotion detection
        features.append(float(confidence))
        
        # Fill the vector with the extracted features
        for i, feature in enumerate(features):
            if i < 9:  # Use first 9 positions for features
                emotion_vector[i] = float(feature)
        
        # One-hot encoding for emotion type (positions 9-16)
        # Thể hiện cảm xúc từ vui vẻ đến trung tính
        emotion_order = ["Happy", "Sad", "Angry", "Surprised", "Fearful", "Disgusted", "Neutral", "Unknown"]
        emotion_index = emotion_mapping.get(emotion, 7)  # Default to unknown
        emotion_vector[9 + emotion_index] = 1.0
        
        # Check for NaN values
        emotion_vector = np.nan_to_num(emotion_vector)
        
        # Convert any numpy float types to Python float for database compatibility
        for i in range(len(emotion_vector)):
            emotion_vector[i] = float(emotion_vector[i])
            
        # Print debug info
        print(f"Emotion vector (224x224 optimized):")
        print(f"- Vector type: {type(emotion_vector)}")
        print(f"- First element type: {type(emotion_vector[0])}")
        print(f"- Detected emotion: {emotion} (index: {emotion_index})")
        
    except Exception as e:
        print(f"Error creating emotion vector: {e}")
    
    return emotion_vector