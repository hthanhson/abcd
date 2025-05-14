import cv2
import numpy as np
import math
import hashlib

# Bộ nhớ đệm toàn cục để lưu trữ kết quả dựa trên hash của ảnh
_gender_cache = {}

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

def logistic_transition(value, low, high, steepness=10):
    """Compute a smooth transition between 0 and 1 using a logistic function."""
    mid = float((low + high) / 2)
    x = float(steepness * (value - mid) / (high - low))
    # Sử dụng math module thay vì numpy để tránh trả về numpy.float64
    try:
        return float(1 / (1 + math.exp(-x)))
    except (OverflowError, ValueError):
        # Xử lý trường hợp đặc biệt để tránh lỗi
        if x > 0:
            return 1.0
        else:
            return 0.0

def detect_gender(face_image, force_recalculate=False):
    """
    Detect gender from a face image with balanced feature analysis.
    Optimized for 224x224 input images, with improved consistency.
    
    Args:
        face_image: Input face image
        force_recalculate: Force recalculation even if image is in cache
    
    Returns:
        tuple: (gender, confidence) - detected gender and confidence level
    """
    # Input validation
    if face_image is None or face_image.size == 0:
        print("Invalid image in gender detection, defaulting to Man")
        return "Man", 0.6
    
    # Kiểm tra bộ nhớ đệm nếu không bị bắt buộc tính toán lại
    if not force_recalculate:
        img_hash = get_image_hash(face_image)
        if img_hash in _gender_cache:
            cached_result = _gender_cache[img_hash]
            print(f"Using cached gender result: {cached_result[0]} (confidence: {cached_result[1]:.2f})")
            return cached_result
    
    try:
        # Ensure image is in grayscale format
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image.copy()
        
        # Save original dimensions before resizing
        original_h, original_w = gray.shape
        original_face_ratio = float(original_h) / float(original_w)
        
        # Chuẩn hóa kích thước ảnh
        resized = cv2.resize(gray, (224, 224))
        
        # Cải thiện tiền xử lý ảnh
        # 1. Giảm nhiễu với Gaussian blur nhẹ
        blurred = cv2.GaussianBlur(resized, (3, 3), 0)
        
        # 2. Tăng cường độ tương phản với CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # 3. Chuẩn hóa độ sáng
        normalized = cv2.normalize(enhanced, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        
        # Calculate features
        h, w = normalized.shape
        
        # 1. Face shape ratio (height to width) - using original dimensions
        face_ratio = original_face_ratio
        
        # 2. Jaw edge intensity - optimized for 224x224
        jaw_region = normalized[int(h*0.75):h, :]
        jaw_edges = cv2.Sobel(jaw_region, cv2.CV_64F, 1, 0) if jaw_region.size > 0 else np.zeros((1, 1))
        jaw_edge_intensity = float(np.mean(np.abs(jaw_edges)))
        
        # 3. Skin smoothness - using larger kernel for 224x224
        texture_kernel = np.ones((7, 7), np.float32) / 49  # Larger kernel for higher resolution
        smoothed = cv2.filter2D(normalized, -1, texture_kernel)
        texture_diff = cv2.absdiff(normalized, smoothed)
        smoothness = float(1.0 - (np.mean(texture_diff) / 255.0))
        
        # 4. Eye region edge intensity - optimized regions for 224x224
        eye_region = normalized[int(h*0.25):int(h*0.45), :]
        eye_edges = cv2.Sobel(eye_region, cv2.CV_64F, 1, 1) if eye_region.size > 0 else np.zeros((1, 1))
        eye_edge_intensity = float(np.mean(np.abs(eye_edges)))
        
        # 5. Cheek smoothness - optimized regions for 224x224
        left_cheek = normalized[int(h*0.4):int(h*0.7), int(w*0.1):int(w*0.35)]
        right_cheek = normalized[int(h*0.4):int(h*0.7), int(w*0.65):int(w*0.9)]
        left_cheek_texture = cv2.absdiff(left_cheek, cv2.filter2D(left_cheek, -1, texture_kernel)) if left_cheek.size > 0 else np.zeros((1, 1))
        right_cheek_texture = cv2.absdiff(right_cheek, cv2.filter2D(right_cheek, -1, texture_kernel)) if right_cheek.size > 0 else np.zeros((1, 1))
        cheek_smoothness = float(1.0 - ((np.mean(left_cheek_texture) + np.mean(right_cheek_texture)) / (2 * 255.0)))
        
        # 6. Mouth area roughness - optimized for 224x224
        mouth_area = normalized[int(h*0.65):int(h*0.85), int(w*0.3):int(w*0.7)]
        mouth_area_texture = cv2.absdiff(mouth_area, cv2.filter2D(mouth_area, -1, texture_kernel)) if mouth_area.size > 0 else np.zeros((1, 1))
        mouth_area_roughness = float(np.mean(mouth_area_texture) / 255.0)
        
        # 7. Eyebrow thickness and arch - better threshold for 224x224
        eyebrow_region = normalized[int(h*0.15):int(h*0.25), :]
        _, eyebrow_thresh = cv2.threshold(eyebrow_region, 110, 255, cv2.THRESH_BINARY) if eyebrow_region.size > 0 else (None, np.zeros((1, 1)))
        eyebrow_thickness = float(np.sum(eyebrow_thresh == 0) / eyebrow_thresh.size) if eyebrow_thresh is not None and eyebrow_thresh.size > 0 else 0.0
        
        # 8. Forehead height - optimized for 224x224
        forehead_height = float(h*0.15) / float(h)  # Proportion of face that is forehead
        
        # 9. Face symmetry analysis - improved for 224x224
        left_half = normalized[:, :int(w/2)]
        right_half = normalized[:, int(w/2):]
        right_half_flipped = cv2.flip(right_half, 1)  # Lật ngang để so sánh với nửa trái
        
        # Đảm bảo kích thước giống nhau
        if left_half.shape[1] > right_half_flipped.shape[1]:
            left_half = left_half[:, :right_half_flipped.shape[1]]
        elif right_half_flipped.shape[1] > left_half.shape[1]:
            right_half_flipped = right_half_flipped[:, :left_half.shape[1]]
            
        # Tính độ đối xứng
        symmetry_diff = cv2.absdiff(left_half, right_half_flipped)
        face_symmetry = float(1.0 - (np.mean(symmetry_diff) / 255.0))
        
        # 10. Thêm: Phát hiện đường viền hàm (quan trọng cho nam giới)
        jawline_region = normalized[int(h*0.7):int(h*0.9), :]
        jawline_edges_x = cv2.Sobel(jawline_region, cv2.CV_64F, 1, 0) if jawline_region.size > 0 else np.zeros((1, 1))
        jawline_edges_y = cv2.Sobel(jawline_region, cv2.CV_64F, 0, 1) if jawline_region.size > 0 else np.zeros((1, 1))
        jawline_intensity = float(np.mean(np.sqrt(jawline_edges_x**2 + jawline_edges_y**2)))
        
        # 11. Nose bridge sharpness (more prominent in males)
        nose_bridge = normalized[int(h*0.3):int(h*0.5), int(w*0.45):int(w*0.55)]
        if nose_bridge.size > 0:
            nose_bridge_edges = cv2.Sobel(nose_bridge, cv2.CV_64F, 0, 1)
            nose_bridge_sharpness = float(np.mean(np.abs(nose_bridge_edges)))
        else:
            nose_bridge_sharpness = 0.0
        
        # 12. Chin prominence
        chin_region = normalized[int(h*0.85):h, int(w*0.35):int(w*0.65)]
        if chin_region.size > 0:
            chin_edges = cv2.Sobel(chin_region, cv2.CV_64F, 0, 1)
            chin_prominence = float(np.mean(np.abs(chin_edges)))
        else:
            chin_prominence = 0.0
        
        # Print feature values for debugging
        print(f"Gender detection metrics (224x224 optimized):")
        print(f"- Face ratio (original): {face_ratio:.2f}")
        print(f"- Jaw edge intensity: {jaw_edge_intensity:.2f}")
        print(f"- Jawline intensity: {jawline_intensity:.2f}")
        print(f"- Smoothness: {smoothness:.2f}")
        print(f"- Eye edge intensity: {eye_edge_intensity:.2f}")
        print(f"- Cheek smoothness: {cheek_smoothness:.2f}")
        print(f"- Mouth area roughness: {mouth_area_roughness:.2f}")
        print(f"- Eyebrow thickness: {eyebrow_thickness:.2f}")
        print(f"- Forehead height: {forehead_height:.2f}")
        print(f"- Face symmetry: {face_symmetry:.2f}")
        print(f"- Nose bridge sharpness: {nose_bridge_sharpness:.2f}")
        print(f"- Chin prominence: {chin_prominence:.2f}")
        
        # Initialize scores
        male_features = 0.0
        female_features = 0.0
        
        # Định nghĩa trọng số cho các tính năng - cân bằng hơn và tối ưu cho 224x224
        weights = {
            "face_ratio": 0.08,         # Giảm trọng số vì tỷ lệ mặt có thể bị ảnh hưởng
            "jaw_edge": 0.10,           # Quan trọng cho nam giới
            "jawline": 0.10,            # Tính năng mới, quan trọng cho nam giới
            "smoothness": 0.10,         # Quan trọng cho nữ giới
            "eye_edge": 0.08,           # Tính năng trung tính
            "cheek_smoothness": 0.10,   # Quan trọng cho nữ giới
            "mouth_roughness": 0.06,    # Giảm xuống vì ít ổn định
            "eyebrow_thickness": 0.10,  # Quan trọng cho sự phân biệt
            "forehead_height": 0.08,    # Khá ổn định
            "face_symmetry": 0.06,      # Giảm xuống vì ít quan trọng
            "nose_bridge": 0.08,        # Quan trọng cho nam giới
            "chin_prominence": 0.06     # Tính năng bổ sung
        }
        
        # 1. Face ratio - adjusted thresholds
        if face_ratio < 0.92:  # Lower threshold
            male_features += float(weights["face_ratio"])
            print(f"- Face ratio indicates Man (+{weights['face_ratio']:.2f})")
        elif face_ratio > 1.08:  # Higher threshold
            female_features += float(weights["face_ratio"])
            print(f"- Face ratio indicates Woman (+{weights['face_ratio']:.2f})")
        else:
            female_ratio = float(logistic_transition(face_ratio, 0.92, 1.08))
            male_ratio = float(1.0 - female_ratio)
            male_features += float(weights["face_ratio"] * male_ratio)
            female_features += float(weights["face_ratio"] * female_ratio)
            print(f"- Face ratio in middle range: Man +{weights['face_ratio'] * male_ratio:.2f}, Woman +{weights['face_ratio'] * female_ratio:.2f}")
        
        # 2. Jaw edge intensity
        jaw_edge_norm = float(jaw_edge_intensity / 80.0)  # Normalize to 0-1
        if jaw_edge_norm > 0.65:  # Giảm ngưỡng để tăng độ nhạy
            male_features += float(weights["jaw_edge"])
            print(f"- Jaw edge intensity indicates Man (+{weights['jaw_edge']:.2f})")
        elif jaw_edge_norm < 0.5:
            female_features += float(weights["jaw_edge"])
            print(f"- Jaw edge intensity indicates Woman (+{weights['jaw_edge']:.2f})")
        else:
            female_ratio = float(1.0 - logistic_transition(jaw_edge_norm, 0.5, 0.65))
            male_ratio = float(1.0 - female_ratio)
            male_features += float(weights["jaw_edge"] * male_ratio)
            female_features += float(weights["jaw_edge"] * female_ratio)
            print(f"- Jaw edge in middle range: Man +{weights['jaw_edge'] * male_ratio:.2f}, Woman +{weights['jaw_edge'] * female_ratio:.2f}")

        # 2b. Jawline intensity (new feature)
        jawline_norm = float(jawline_intensity / 100.0)  # Normalize to 0-1
        if jawline_norm > 0.65:
            male_features += float(weights["jawline"])
            print(f"- Jawline intensity indicates Man (+{weights['jawline']:.2f})")
        elif jawline_norm < 0.5:
            female_features += float(weights["jawline"])
            print(f"- Jawline intensity indicates Woman (+{weights['jawline']:.2f})")
        else:
            female_ratio = float(1.0 - logistic_transition(jawline_norm, 0.5, 0.65))
            male_ratio = float(1.0 - female_ratio)
            male_features += float(weights["jawline"] * male_ratio)
            female_features += float(weights["jawline"] * female_ratio)
            print(f"- Jawline in middle range: Man +{weights['jawline'] * male_ratio:.2f}, Woman +{weights['jawline'] * female_ratio:.2f}")
        
        # 3. Skin smoothness
        if smoothness < 0.65:
            male_features += float(weights["smoothness"])
            print(f"- Smoothness indicates Man (+{weights['smoothness']:.2f})")
        elif smoothness > 0.75:
            female_features += float(weights["smoothness"])
            print(f"- Smoothness indicates Woman (+{weights['smoothness']:.2f})")
        else:
            female_ratio = float(logistic_transition(smoothness, 0.65, 0.75))
            male_ratio = float(1.0 - female_ratio)
            male_features += float(weights["smoothness"] * male_ratio)
            female_features += float(weights["smoothness"] * female_ratio)
            print(f"- Smoothness in middle range: Man +{weights['smoothness'] * male_ratio:.2f}, Woman +{weights['smoothness'] * female_ratio:.2f}")
        
        # 4. Eye edge intensity
        eye_edge_norm = float(eye_edge_intensity / 80.0)  # Normalize to 0-1
        if eye_edge_norm > 0.7:
            male_features += float(weights["eye_edge"])
            print(f"- Eye edge intensity indicates Man (+{weights['eye_edge']:.2f})")
        elif eye_edge_norm < 0.5:
            female_features += float(weights["eye_edge"])
            print(f"- Eye edge intensity indicates Woman (+{weights['eye_edge']:.2f})")
        else:
            female_ratio = float(1.0 - logistic_transition(eye_edge_norm, 0.5, 0.7))
            male_ratio = float(1.0 - female_ratio)
            male_features += float(weights["eye_edge"] * male_ratio)
            female_features += float(weights["eye_edge"] * female_ratio)
            print(f"- Eye edge in middle range: Man +{weights['eye_edge'] * male_ratio:.2f}, Woman +{weights['eye_edge'] * female_ratio:.2f}")
        
        # 5. Cheek smoothness
        if cheek_smoothness < 0.65:
            male_features += float(weights["cheek_smoothness"])
            print(f"- Cheek smoothness indicates Man (+{weights['cheek_smoothness']:.2f})")
        elif cheek_smoothness > 0.75:
            female_features += float(weights["cheek_smoothness"])
            print(f"- Cheek smoothness indicates Woman (+{weights['cheek_smoothness']:.2f})")
        else:
            female_ratio = float(logistic_transition(cheek_smoothness, 0.65, 0.75))
            male_ratio = float(1.0 - female_ratio)
            male_features += float(weights["cheek_smoothness"] * male_ratio)
            female_features += float(weights["cheek_smoothness"] * female_ratio)
            print(f"- Cheek smoothness in middle range: Man +{weights['cheek_smoothness'] * male_ratio:.2f}, Woman +{weights['cheek_smoothness'] * female_ratio:.2f}")
        
        # 6. Mouth area roughness
        if mouth_area_roughness < 0.65:
            male_features += float(weights["mouth_roughness"])
            print(f"- Mouth area roughness indicates Man (+{weights['mouth_roughness']:.2f})")
        elif mouth_area_roughness > 0.75:
            female_features += float(weights["mouth_roughness"])
            print(f"- Mouth area roughness indicates Woman (+{weights['mouth_roughness']:.2f})")
        else:
            female_ratio = float(logistic_transition(mouth_area_roughness, 0.65, 0.75))
            male_ratio = float(1.0 - female_ratio)
            male_features += float(weights["mouth_roughness"] * male_ratio)
            female_features += float(weights["mouth_roughness"] * female_ratio)
            print(f"- Mouth area roughness in middle range: Man +{weights['mouth_roughness'] * male_ratio:.2f}, Woman +{weights['mouth_roughness'] * female_ratio:.2f}")
        
        # 7. Eyebrow thickness
        if eyebrow_thickness < 0.65:
            male_features += float(weights["eyebrow_thickness"])
            print(f"- Eyebrow thickness indicates Man (+{weights['eyebrow_thickness']:.2f})")
        elif eyebrow_thickness > 0.75:
            female_features += float(weights["eyebrow_thickness"])
            print(f"- Eyebrow thickness indicates Woman (+{weights['eyebrow_thickness']:.2f})")
        else:
            female_ratio = float(logistic_transition(eyebrow_thickness, 0.65, 0.75))
            male_ratio = float(1.0 - female_ratio)
            male_features += float(weights["eyebrow_thickness"] * male_ratio)
            female_features += float(weights["eyebrow_thickness"] * female_ratio)
            print(f"- Eyebrow thickness in middle range: Man +{weights['eyebrow_thickness'] * male_ratio:.2f}, Woman +{weights['eyebrow_thickness'] * female_ratio:.2f}")
        
        # 8. Forehead height
        if forehead_height < 0.18:
            male_features += float(weights["forehead_height"])
            print(f"- Forehead height indicates Man (+{weights['forehead_height']:.2f})")
        elif forehead_height > 0.28:
            female_features += float(weights["forehead_height"])
            print(f"- Forehead height indicates Woman (+{weights['forehead_height']:.2f})")
        else:
            female_ratio = float(logistic_transition(forehead_height, 0.18, 0.28))
            male_ratio = float(1.0 - female_ratio)
            male_features += float(weights["forehead_height"] * male_ratio)
            female_features += float(weights["forehead_height"] * female_ratio)
            print(f"- Forehead height in middle range: Man +{weights['forehead_height'] * male_ratio:.2f}, Woman +{weights['forehead_height'] * female_ratio:.2f}")
        
        # 9. Face symmetry
        if face_symmetry < 0.65:
            male_features += float(weights["face_symmetry"])
            print(f"- Face symmetry indicates Man (+{weights['face_symmetry']:.2f})")
        elif face_symmetry > 0.85:
            female_features += float(weights["face_symmetry"])
            print(f"- Face symmetry indicates Woman (+{weights['face_symmetry']:.2f})")
        else:
            female_ratio = float(logistic_transition(face_symmetry, 0.65, 0.85))
            male_ratio = float(1.0 - female_ratio)
            male_features += float(weights["face_symmetry"] * male_ratio)
            female_features += float(weights["face_symmetry"] * female_ratio)
            print(f"- Face symmetry in middle range: Man +{weights['face_symmetry'] * male_ratio:.2f}, Woman +{weights['face_symmetry'] * female_ratio:.2f}")
        
        # 10. Nose bridge sharpness
        nose_bridge_norm = float(nose_bridge_sharpness / 100.0)  # Normalize to 0-1
        if nose_bridge_norm > 0.7:
            male_features += float(weights["nose_bridge"])
            print(f"- Nose bridge sharpness indicates Man (+{weights['nose_bridge']:.2f})")
        elif nose_bridge_norm < 0.5:
            female_features += float(weights["nose_bridge"])
            print(f"- Nose bridge sharpness indicates Woman (+{weights['nose_bridge']:.2f})")
        else:
            female_ratio = float(1.0 - logistic_transition(nose_bridge_norm, 0.5, 0.7))
            male_ratio = float(1.0 - female_ratio)
            male_features += float(weights["nose_bridge"] * male_ratio)
            female_features += float(weights["nose_bridge"] * female_ratio)
            print(f"- Nose bridge sharpness in middle range: Man +{weights['nose_bridge'] * male_ratio:.2f}, Woman +{weights['nose_bridge'] * female_ratio:.2f}")
        
        # 11. Chin prominence
        chin_norm = float(chin_prominence / 100.0)  # Normalize to 0-1
        if chin_norm > 0.7:
            male_features += float(weights["chin_prominence"])
            print(f"- Chin prominence indicates Man (+{weights['chin_prominence']:.2f})")
        elif chin_norm < 0.5:
            female_features += float(weights["chin_prominence"])
            print(f"- Chin prominence indicates Woman (+{weights['chin_prominence']:.2f})")
        else:
            female_ratio = float(1.0 - logistic_transition(chin_norm, 0.5, 0.7))
            male_ratio = float(1.0 - female_ratio)
            male_features += float(weights["chin_prominence"] * male_ratio)
            female_features += float(weights["chin_prominence"] * female_ratio)
            print(f"- Chin prominence in middle range: Man +{weights['chin_prominence'] * male_ratio:.2f}, Woman +{weights['chin_prominence'] * female_ratio:.2f}")
        
        # Determine gender based on feature scores
        total_score = male_features + female_features
        male_score = male_features / total_score if total_score > 0 else 0.5
        female_score = female_features / total_score if total_score > 0 else 0.5
        
        # Print scores
        print(f"- Male score: {male_score:.2f}, Female score: {female_score:.2f}")
        
        # Thêm: Xác định giới tính một cách chắc chắn
        score_difference = abs(male_score - female_score)
        
        # Determine gender with confidence
        if male_score > female_score:
            gender = "Man"
            confidence = male_score
        else:
            gender = "Woman"
            confidence = female_score
        
        # Yêu cầu độ chênh lệch tối thiểu để đảm bảo kết quả chắc chắn
        # Thay thế yếu tố ngẫu nhiên với một quy tắc ổn định hơn
        if score_difference < 0.1:  # Nếu điểm nam và nữ quá gần nhau
            # Dựa vào các đặc điểm mạnh và nhất quán hơn
            nose_chin_jawline = (nose_bridge_norm + chin_norm + jawline_norm) / 3
            if nose_chin_jawline > 0.6:
                gender = "Man"
                confidence = 0.6
                print(f"- Low confidence difference ({score_difference:.2f}), using structural features: {gender}")
            else:
                gender = "Woman"
                confidence = 0.6
                print(f"- Low confidence difference ({score_difference:.2f}), using structural features: {gender}")
        else:
            confidence = min(confidence, 0.98)  # Cap maximum confidence
            print(f"- Final gender: {gender} (confidence: {confidence:.2f})")
        
        # Lưu kết quả vào bộ nhớ đệm
        img_hash = get_image_hash(face_image)
        _gender_cache[img_hash] = (gender, confidence)
        
        return gender, confidence
        
    except Exception as e:
        print(f"Error in gender detection: {e}")
        # Return a consistent default result on error
        return "Man", 0.6

def get_gender_vector(face_image, vector_length=16):
    """
    Create feature vector for gender from face image - optimized for 224x224
    
    Args:
        face_image: Face image
        vector_length: Length of output vector
        
    Returns:
        ndarray: Gender feature vector (16-dimensional)
    """
    # Initialize feature vector - sử dụng Python list
    gender_vector = [0.0] * vector_length
    
    # Validate input
    if face_image is None or face_image.size == 0:
        return np.array(gender_vector, dtype=float)
    
    try:
        # Detect gender and confidence - sử dụng bộ nhớ đệm
        gender, confidence = detect_gender(face_image)
        confidence = float(confidence)
        
        # Define gender indices for one-hot encoding
        gender_mapping = {
            "Man": 0,
            "Woman": 1,
            "Unknown": 2
        }
        gender_index = gender_mapping.get(gender, 2)  # Default to unknown
        
        # Create a copy for feature extraction
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image.copy()
        
        # Resize for consistency - use 224x224 for better feature extraction
        gray = cv2.resize(gray, (224, 224))
        
        # Cải thiện tiền xử lý ảnh
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        gray = clahe.apply(blurred)
        
        # Create feature list
        features = []
        
        # 1. Face ratio (height to width)
        h, w = gray.shape
        face_ratio = float(h) / float(w)
        features.append(float(min(face_ratio, 1.5)))  # Normalized
        
        # Define regions for feature extraction - optimized for 224x224
        jaw_region = gray[int(h*0.75):h, :]
        eye_region = gray[int(h*0.25):int(h*0.45), :]
        eyebrow_region = gray[int(h*0.15):int(h*0.25), :]
        nose_bridge = gray[int(h*0.3):int(h*0.5), int(w*0.45):int(w*0.55)]
        
        # 2. Jaw edge strength - optimized for 224x224
        jaw_edges = cv2.Sobel(jaw_region, cv2.CV_64F, 1, 0) if jaw_region.size > 0 else np.zeros((1, 1))
        jaw_edge_mean = float(np.mean(np.abs(jaw_edges)) / 150.0)  # Adjusted normalization for 224x224
        features.append(float(min(jaw_edge_mean, 1.0)))
        
        # 3. Eye edge strength - optimized for 224x224
        eye_edges = cv2.Sobel(eye_region, cv2.CV_64F, 1, 1) if eye_region.size > 0 else np.zeros((1, 1))
        eye_edge_mean = float(np.mean(np.abs(eye_edges)) / 150.0)  # Adjusted normalization for 224x224
        features.append(float(min(eye_edge_mean, 1.0)))
        
        # 4. Texture - optimized for 224x224
        texture_kernel = np.ones((7, 7), np.float32) / 49  # Larger kernel for 224x224
        smoothed = cv2.filter2D(gray, -1, texture_kernel)
        texture_diff = cv2.absdiff(gray, smoothed)
        texture_value = float(np.mean(texture_diff) / 255.0)  # Chuyển đổi sang Python float
        features.append(texture_value)
        
        # 5. Eyebrow thickness - optimized for 224x224
        _, eyebrow_thresh = cv2.threshold(eyebrow_region, 110, 255, cv2.THRESH_BINARY) if eyebrow_region.size > 0 else (None, np.zeros((1, 1)))
        eyebrow_thickness = float(np.sum(eyebrow_thresh == 0) / eyebrow_thresh.size) if eyebrow_thresh is not None and eyebrow_thresh.size > 0 else 0.0
        features.append(float(min(eyebrow_thickness * 3.0, 1.0)))  # Normalize and scale
        
        # 6. Confidence
        features.append(confidence)
        
        # 7. Gender balance (soft indicator)
        gender_balance = 0.0 if gender == "Man" else 1.0
        features.append(gender_balance)
        
        # 8. New feature: Nose bridge sharpness
        if nose_bridge.size > 0:
            nose_bridge_edges = cv2.Sobel(nose_bridge, cv2.CV_64F, 0, 1)
            nose_bridge_sharpness = float(np.mean(np.abs(nose_bridge_edges)) / 150.0)  # Normalized for 224x224
            features.append(float(min(nose_bridge_sharpness, 1.0)))
        else:
            features.append(0.5)  # Default value
        
        # Fill vector
        for i, feature in enumerate(features):
            if i < 8:
                gender_vector[i] = float(feature)
        
        # One-hot encoding
        gender_vector[8 + gender_index] = 1.0
        
        # Additional features
        gender_vector[11] = float(jaw_edge_mean * confidence)
        gender_vector[12] = float((1.0 - texture_value) * confidence)
        gender_vector[13] = gender_balance
        gender_vector[14] = 1.0 - gender_balance
        gender_vector[15] = float(confidence * (1.0 if gender_index < 2 else 0.5))
        
        # Handle NaN
        for i in range(len(gender_vector)):
            if math.isnan(float(gender_vector[i])):
                gender_vector[i] = 0.0
        
    except Exception as e:
        print(f"Error creating gender vector: {e}")
    
    # Kiểm tra cuối cùng để đảm bảo tất cả là float thuần Python
    for i in range(len(gender_vector)):
        # Chuyển đổi bất kỳ kiểu dữ liệu nào thành float Python chuẩn
        gender_vector[i] = float(gender_vector[i])
    
    # Tạo mảng NumPy từ danh sách Python, với kiểu dữ liệu float Python
    numpy_vector = np.array(gender_vector, dtype=float)
    
    # In thông tin kiểm tra để debug
    print(f"Gender vector (224x224 optimized) type: {type(numpy_vector)}")
    print(f"First element type: {type(numpy_vector[0])}")
    print(f"NumPy dtype: {numpy_vector.dtype}")
    
    return numpy_vector

def clear_gender_cache():
    """
    Xóa bộ nhớ đệm kết quả giới tính
    """
    global _gender_cache
    _gender_cache = {}
    print("Gender detection cache cleared")