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
        return "Man", 0.75
    
    # Kiểm tra bộ nhớ đệm nếu không bị bắt buộc tính toán lại
    if not force_recalculate:
        img_hash = get_image_hash(face_image)
        if img_hash in _gender_cache:
            cached_result = _gender_cache[img_hash]
            print(f"Using cached gender result: {cached_result[0]} (confidence: {cached_result[1]:.2f})")
            return cached_result
    
    try:
        # Ensure image is in proper format
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            color = face_image.copy()
        else:
            gray = face_image.copy()
            # Convert to color for some features
            color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Save original dimensions before resizing
        original_h, original_w = gray.shape[:2]
        original_face_ratio = float(original_h) / float(original_w)
        
        # Chuẩn hóa kích thước ảnh - tối ưu cho 224x224
        resized_gray = cv2.resize(gray, (224, 224))
        resized_color = cv2.resize(color, (224, 224))
        
        # Cải thiện tiền xử lý ảnh
        # 1. Giảm nhiễu với bilateral filter - bảo toàn cạnh tốt hơn
        denoised = cv2.bilateralFilter(resized_gray, 5, 25, 25)
        
        # 2. Tăng cường độ tương phản với CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 3. Chuẩn hóa độ sáng
        normalized = cv2.normalize(enhanced, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        
        # Calculate features
        h, w = normalized.shape
        face_ratio = original_face_ratio
        
        # Tính toán các đặc trưng được tối ưu cho ảnh 224x224
        # ----------------------------------------------------
        
        # 1. Face shape ratio - tỷ lệ khuôn mặt
        face_ratio_score = logistic_transition(face_ratio, 0.9, 1.1)
        print(f"- Face ratio: {face_ratio:.2f}, score: {face_ratio_score:.2f}")
        
        # 2. Texture analysis - phân tích kết cấu/độ mịn
        # Sử dụng bộ lọc Gabor để phát hiện đặc trưng kết cấu
        ksize = 9
        sigma = 3.0
        theta = 0
        lambd = 10.0
        gamma = 0.5
        
        gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        gabor_img = cv2.filter2D(normalized, cv2.CV_8UC3, gabor_kernel)
        texture_diff = cv2.absdiff(normalized, gabor_img)
        smoothness = float(1.0 - (np.mean(texture_diff) / 128.0))
        smoothness_score = logistic_transition(smoothness, 0.65, 0.85)
        print(f"- Smoothness: {smoothness:.2f}, score: {smoothness_score:.2f}")
        
        # 3. Cheek analysis - phân tích vùng má
        # Điều chỉnh vùng má để phù hợp với ảnh 224x224
        left_cheek = normalized[int(h*0.35):int(h*0.65), int(w*0.1):int(w*0.3)]
        right_cheek = normalized[int(h*0.35):int(h*0.65), int(w*0.7):int(w*0.9)]
        
        cheek_kernel = np.ones((5, 5), np.float32) / 25
        
        if left_cheek.size > 0 and right_cheek.size > 0:
            left_cheek_texture = cv2.absdiff(left_cheek, cv2.filter2D(left_cheek, -1, cheek_kernel))
            right_cheek_texture = cv2.absdiff(right_cheek, cv2.filter2D(right_cheek, -1, cheek_kernel))
            cheek_smoothness = float(1.0 - ((np.mean(left_cheek_texture) + np.mean(right_cheek_texture)) / (2 * 128.0)))
        else:
            cheek_smoothness = 0.5
            
        cheek_score = logistic_transition(cheek_smoothness, 0.6, 0.8)
        print(f"- Cheek smoothness: {cheek_smoothness:.2f}, score: {cheek_score:.2f}")
        
        # 4. Eye region analysis - phân tích vùng mắt
        # Tối ưu vùng mắt cho ảnh 224x224
        eye_region = normalized[int(h*0.2):int(h*0.45), int(w*0.15):int(w*0.85)]
        
        if eye_region.size > 0:
            # Sử dụng phân tích độ tương phản và gradient để phát hiện mắt
            eye_gradient_x = cv2.Sobel(eye_region, cv2.CV_64F, 1, 0, ksize=3)
            eye_gradient_y = cv2.Sobel(eye_region, cv2.CV_64F, 0, 1, ksize=3)
            eye_gradient_mag = np.sqrt(eye_gradient_x**2 + eye_gradient_y**2)
            eye_contrast = np.std(eye_region)
            eye_gradient_mean = np.mean(eye_gradient_mag)
            
            # Kết hợp các đặc trưng mắt
            eye_feature = (eye_contrast / 50.0) * 0.7 + (eye_gradient_mean / 30.0) * 0.3
            eye_feature = min(eye_feature, 1.5)  # Giới hạn giá trị tối đa
        else:
            eye_feature = 0.5
            
        eye_score = logistic_transition(eye_feature, 0.5, 1.0)
        print(f"- Eye feature: {eye_feature:.2f}, score: {eye_score:.2f}")
        
        # 5. Face symmetry analysis - phân tích độ đối xứng
        # Tối ưu cho ảnh 224x224
        left_half = normalized[:, :int(w/2)]
        right_half = normalized[:, int(w/2):]
        right_half_flipped = cv2.flip(right_half, 1)
        
        # Đảm bảo kích thước giống nhau
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        left_half_matched = left_half[:, :min_width]
        right_half_flipped_matched = right_half_flipped[:, :min_width]
            
        # Tính độ đối xứng
        symmetry_diff = cv2.absdiff(left_half_matched, right_half_flipped_matched)
        face_symmetry = float(1.0 - (np.mean(symmetry_diff) / 128.0))
        symmetry_score = logistic_transition(face_symmetry, 0.7, 0.9)
        print(f"- Face symmetry: {face_symmetry:.2f}, score: {symmetry_score:.2f}")
        
        # 6. Thêm: Jawline analysis - phân tích đường hàm
        jaw_region = normalized[int(h*0.7):int(h*0.9), int(w*0.2):int(w*0.8)]
        
        if jaw_region.size > 0:
            # Phát hiện cạnh với Canny
            jaw_edges = cv2.Canny(jaw_region, 50, 150)
            jaw_edge_density = np.mean(jaw_edges) / 255.0
            
            # Tính toán gradient cho đường hàm
            jaw_gradient_y = cv2.Sobel(jaw_region, cv2.CV_64F, 0, 1, ksize=3)
            jaw_gradient_strength = np.mean(np.abs(jaw_gradient_y))
            
            # Kết hợp các đặc trưng jawline
            jaw_feature = (jaw_edge_density * 0.4) + (jaw_gradient_strength / 50.0 * 0.6)
        else:
            jaw_feature = 0.5
            
        # Đàn ông thường có đường hàm mạnh hơn
        jaw_score = 1.0 - logistic_transition(jaw_feature, 0.3, 0.7)
        print(f"- Jaw feature: {jaw_feature:.2f}, score: {jaw_score:.2f}")
        
        # 7. Thêm: Forehead analysis - phân tích vùng trán
        forehead_region = normalized[int(h*0.05):int(h*0.2), int(w*0.25):int(w*0.75)]
        
        if forehead_region.size > 0:
            forehead_texture = cv2.absdiff(forehead_region, cv2.filter2D(forehead_region, -1, cheek_kernel))
            forehead_smoothness = float(1.0 - (np.mean(forehead_texture) / 128.0))
        else:
            forehead_smoothness = 0.5
            
        forehead_score = logistic_transition(forehead_smoothness, 0.6, 0.8)
        print(f"- Forehead smoothness: {forehead_smoothness:.2f}, score: {forehead_score:.2f}")
        
        # ----------- Tính toán điểm tổng hợp và quyết định -------------
        
        # Trọng số cho các đặc trưng khác nhau
        feature_weights = {
            'face_ratio': 0.15,
            'smoothness': 0.20,
            'cheek': 0.15,
            'eye': 0.15,
            'symmetry': 0.10,
            'jaw': 0.15,
            'forehead': 0.10
        }
        
        # Tính điểm tổng hợp (càng cao càng nghiêng về nữ)
        woman_composite_score = (
            face_ratio_score * feature_weights['face_ratio'] +
            smoothness_score * feature_weights['smoothness'] +
            cheek_score * feature_weights['cheek'] +
            eye_score * feature_weights['eye'] +
            symmetry_score * feature_weights['symmetry'] +
            jaw_score * feature_weights['jaw'] +
            forehead_score * feature_weights['forehead']
        )
        
        print(f"- Woman composite score: {woman_composite_score:.4f}")
        
        # Sử dụng ngưỡng động để quyết định, được tối ưu cho ảnh 224x224
        # Ngưỡng mặc định được điều chỉnh để phù hợp với tỷ lệ thực tế
        threshold = 0.55  # Ngưỡng cơ bản cho phân loại
        
        # Quyết định và tính độ tin cậy
        if woman_composite_score >= threshold:
            gender = "Woman"
            # Độ tin cậy tỷ lệ với mức độ vượt ngưỡng
            confidence = 0.7 + min(0.25, (woman_composite_score - threshold) * 2)
        else:
            gender = "Man"
            # Độ tin cậy tỷ lệ với khoảng cách từ ngưỡng
            confidence = 0.7 + min(0.25, (threshold - woman_composite_score) * 2)
        
        # Giới hạn độ tin cậy tối đa
        confidence = min(confidence, 0.95)
        
        print(f"- Final decision: {gender} (confidence: {confidence:.2f})")
        
        # Lưu kết quả vào bộ nhớ đệm
        img_hash = get_image_hash(face_image)
        _gender_cache[img_hash] = (gender, confidence)
        
        return gender, confidence
        
    except Exception as e:
        print(f"Error in gender detection: {e}")
        # Return a consistent default result on error
        return "Man", 0.7

def get_gender_vector(face_image, vector_length=16):
    """
    Create feature vector for gender from face image
    
    Args:
        face_image: Face image
        vector_length: Length of output vector
        
    Returns:
        ndarray: Gender feature vector (16-dimensional)
    """
    # Initialize feature vector
    gender_vector = [0.0] * vector_length
    
    # Validate input
    if face_image is None or face_image.size == 0:
        gender_vector[14] = 1.0  # Đánh dấu là Man
        return np.array(gender_vector, dtype=float)
    
    try:
        # Detect gender and confidence
        gender, confidence = detect_gender(face_image)
        confidence = float(confidence)
        
        # Define gender indices for one-hot encoding
        gender_mapping = {
            "Man": 0,
            "Woman": 1,
            "Unknown": 2
        }
        gender_index = gender_mapping.get(gender, 2)
        
        # Create a copy for feature extraction
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image.copy()
        
        # Resize for consistency
        gray = cv2.resize(gray, (224, 224))
        
        # Cải thiện tiền xử lý ảnh
        denoised = cv2.bilateralFilter(gray, 5, 25, 25)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Tính toán một số đặc trưng cơ bản
        h, w = enhanced.shape
        
        # Feature 1: Tỷ lệ khuôn mặt
        face_ratio = float(h) / float(w)
        
        # Feature 2-7: Các đặc trưng được tính toán từ hàm detect_gender
        # Chúng ta chỉ cần tính toán một số đặc trưng chính
        
        # Đánh giá đặc trưng texture
        texture_kernel = np.ones((7, 7), np.float32) / 49
        smoothed = cv2.filter2D(enhanced, -1, texture_kernel)
        texture_diff = cv2.absdiff(enhanced, smoothed)
        smoothness = float(1.0 - (np.mean(texture_diff) / 255.0))
        
        # Đánh giá đặc trưng mắt
        eye_region = enhanced[int(h*0.2):int(h*0.45), :]
        eye_contrast = np.std(eye_region) if eye_region.size > 0 else 0
        eye_feature = min(eye_contrast / 50.0, 1.0)
        
        # Đánh giá đặc trưng đối xứng
        left_half = enhanced[:, :int(w/2)]
        right_half = enhanced[:, int(w/2):]
        right_half_flipped = cv2.flip(right_half, 1)
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        symmetry_diff = cv2.absdiff(left_half[:, :min_width], right_half_flipped[:, :min_width])
        symmetry = float(1.0 - (np.mean(symmetry_diff) / 255.0))
        
        # Tổng hợp các đặc trưng
        features = [
            min(face_ratio, 1.5),    # Tỷ lệ khuôn mặt
            smoothness,              # Độ mịn
            eye_feature,             # Đặc trưng mắt
            symmetry,                # Độ đối xứng
            float(gender_index),     # Chỉ số giới tính
            confidence,              # Độ tin cậy
            1.0 if gender == "Woman" else 0.0,  # Đánh dấu Woman
            1.0 if gender == "Man" else 0.0     # Đánh dấu Man
        ]
        
        # Fill vector
        for i, feature in enumerate(features):
            if i < 8:
                gender_vector[i] = float(feature)
        
        # One-hot encoding
        gender_vector[8 + gender_index] = 1.0
        
        # Additional features
        gender_vector[11] = float(smoothness)  # Độ mịn
        gender_vector[12] = float(eye_feature)  # Đặc trưng mắt
        gender_vector[13] = 1.0 if gender == "Woman" else 0.0  # Đánh dấu Woman
        gender_vector[14] = 0.0 if gender == "Woman" else 1.0  # Đánh dấu Man

        
        # Handle NaN
        for i in range(len(gender_vector)):
            if math.isnan(float(gender_vector[i])):
                gender_vector[i] = 0.0
        
    except Exception as e:
        print(f"Error creating gender vector: {e}")
        gender_vector[14] = 1.0  # Mặc định là Man

    
    # Kiểm tra cuối cùng để đảm bảo tất cả là float thuần Python
    for i in range(len(gender_vector)):
        gender_vector[i] = float(gender_vector[i])
    
    # Tạo mảng NumPy
    numpy_vector = np.array(gender_vector, dtype=float)
    
    return numpy_vector

