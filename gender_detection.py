import cv2
import numpy as np
import math
import hashlib

# Global cache to store results based on image hash and age group
_gender_cache = {}

def get_image_hash(image):
    """
    Create a unique hash for the image for caching purposes.
    """
    if image is None or image.size == 0:
        return "empty_image"
    
    small_img = cv2.resize(image, (32, 32))
    if len(small_img.shape) == 3:
        small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    
    img_bytes = small_img.tobytes()
    img_hash = hashlib.md5(img_bytes).hexdigest()
    return img_hash

def detect_gender(face_image, force_recalculate=False, face_encoding=None):
    """
    Detect gender from a face image using face encoding.
    Optimized for 224x224 input images.
    
    Args:
        face_image: Input face image
        force_recalculate: Force recalculation even if image is in cache
        face_encoding: Face encoding vector (128-dim) nếu có, để cải thiện độ chính xác
    
    Returns:
        tuple: (gender, confidence) - detected gender and confidence level
    """
    if face_image is None or face_image.size == 0:
        print("Invalid image in gender detection, defaulting to Man")
        return "Man", 0.70
    
    # Chuyển đổi hình ảnh sang kích thước 224x224 trước khi xử lý
    if face_image.shape[0] != 224 or face_image.shape[1] != 224:
        face_image = cv2.resize(face_image, (224, 224))
    
    # Tạo khóa bộ nhớ đệm dựa trên cả hình ảnh và face_encoding nếu có
    # Nếu face_encoding được cung cấp, tạo một hash từ vector đó
    if not force_recalculate:
        img_hash = get_image_hash(face_image)
        encoding_hash = ""
        if face_encoding is not None:
            # Tạo hash cho face_encoding
            encoding_str = str(face_encoding.tobytes())
            encoding_hash = hashlib.md5(encoding_str.encode()).hexdigest()[:8]
            
        cache_key = (img_hash, encoding_hash)
        if cache_key in _gender_cache:
            cached_result = _gender_cache[cache_key]
            print(f"Using cached gender result: {cached_result[0]} (confidence: {cached_result[1]:.2f})")
            return cached_result
    
    try:
        # Lấy các tham số mặc định
        params = {
            'gender_threshold': 0.65,  # Tăng từ 0.55 lên 0.65
            'bilateral_filter': (5, 25, 25),
            'clahe_params': (3.0, (8, 8)),
            'canny_threshold': (50, 150)
        }
        
        # Chuẩn bị hình ảnh
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            color = face_image.copy()
            hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
        else:
            gray = face_image.copy()
            color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        
        # ===== PHÂN TÍCH ĐẶC TRƯNG =====
        # Cải thiện chất lượng ảnh với bilateral filtering và CLAHE
        b_size, b_sigma1, b_sigma2 = params['bilateral_filter']
        denoised = cv2.bilateralFilter(gray, b_size, b_sigma1, b_sigma2)
        
        clahe_clip, clahe_grid = params['clahe_params']
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        enhanced = clahe.apply(denoised)
        
        h, w = enhanced.shape
        
        # 1. Trích xuất đặc trưng cơ bản từ hình ảnh
        features = {}
        
        # 1.1 Tỷ lệ khuôn mặt
        face_ratio = float(h) / float(w)
        features['face_ratio'] = face_ratio
        
        # 1.2 Phân tích kết cấu da (texture - smoothness)
        texture_kernel = np.ones((7, 7), np.float32) / 49
        smoothed = cv2.filter2D(enhanced, -1, texture_kernel)
        texture_diff = cv2.absdiff(enhanced, smoothed)
        smoothness = float(1.0 - (np.mean(texture_diff) / 255.0))
        features['smoothness'] = smoothness
        
        # 1.3 Phân tích vùng má
        left_cheek = enhanced[int(h*0.35):int(h*0.65), int(w*0.1):int(w*0.3)]
        right_cheek = enhanced[int(h*0.35):int(h*0.65), int(w*0.7):int(w*0.9)]
        cheek_kernel = np.ones((5, 5), np.float32) / 25
        
        if left_cheek.size > 0 and right_cheek.size > 0:
            left_cheek_filtered = cv2.bilateralFilter(left_cheek, 5, 35, 35)
            right_cheek_filtered = cv2.bilateralFilter(right_cheek, 5, 35, 35)
            
            left_cheek_texture = cv2.absdiff(left_cheek, cv2.filter2D(left_cheek_filtered, -1, cheek_kernel))
            right_cheek_texture = cv2.absdiff(right_cheek, cv2.filter2D(right_cheek_filtered, -1, cheek_kernel))
            
            cheek_smoothness = float(1.0 - ((np.mean(left_cheek_texture) + np.mean(right_cheek_texture)) / (2 * 128.0)))
            
            # Phân tích màu sắc vùng má nếu có hình màu
            if len(face_image.shape) == 3:
                hsv_left_cheek = hsv[int(h*0.35):int(h*0.65), int(w*0.1):int(w*0.3)]
                hsv_right_cheek = hsv[int(h*0.35):int(h*0.65), int(w*0.7):int(w*0.9)]
                
                left_saturation = np.mean(hsv_left_cheek[:, :, 1]) if hsv_left_cheek.size > 0 else 0
                right_saturation = np.mean(hsv_right_cheek[:, :, 1]) if hsv_right_cheek.size > 0 else 0
                
                avg_saturation = (left_saturation + right_saturation) / (2 * 255.0)
                features['cheek_saturation'] = avg_saturation
        else:
            cheek_smoothness = 0.5
        features['cheek_smoothness'] = cheek_smoothness
        
        # 1.4 Phân tích vùng mắt
        eye_region = enhanced[int(h*0.2):int(h*0.45), int(w*0.15):int(w*0.85)]
        if eye_region.size > 0:
            eye_gradient_x = cv2.Sobel(eye_region, cv2.CV_64F, 1, 0, ksize=3)
            eye_gradient_y = cv2.Sobel(eye_region, cv2.CV_64F, 0, 1, ksize=3)
            eye_gradient_mag = np.sqrt(eye_gradient_x**2 + eye_gradient_y**2)
            
            eye_contrast = np.std(eye_region)
            eye_gradient_mean = np.mean(eye_gradient_mag)
            
            eye_feature = (eye_contrast / 50.0) * 0.7 + (eye_gradient_mean / 30.0) * 0.3
            
            # Phân tích lông mày
            eyebrow_region = enhanced[int(h*0.15):int(h*0.25), int(w*0.15):int(w*0.85)]
            if eyebrow_region.size > 0:
                eyebrow_gradient_x = cv2.Sobel(eyebrow_region, cv2.CV_64F, 1, 0, ksize=3)
                eyebrow_gradient_y = cv2.Sobel(eyebrow_region, cv2.CV_64F, 0, 1, ksize=3)
                eyebrow_gradient_mag = np.sqrt(eyebrow_gradient_x**2 + eyebrow_gradient_y**2)
                eyebrow_intensity = np.mean(eyebrow_gradient_mag)
                
                if eyebrow_intensity > 40:
                    eye_feature = min(1.5, eye_feature * 1.1)
                
                features['eyebrow_intensity'] = eyebrow_intensity
        else:
            eye_feature = 0.5
        features['eye_feature'] = eye_feature
        
        # 1.5 Phân tích tính đối xứng
        left_half = enhanced[:, :int(w/2)]
        right_half = enhanced[:, int(w/2):]
        right_half_flipped = cv2.flip(right_half, 1)
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        
        if min_width > 0:
            symmetry_diff = cv2.absdiff(left_half[:, :min_width], right_half_flipped[:, :min_width])
            face_symmetry = float(1.0 - (np.mean(symmetry_diff) / 128.0))
        else:
            face_symmetry = 0.5
        features['symmetry'] = face_symmetry
        
        # 1.6 Phân tích đường hàm
        jaw_region = enhanced[int(h*0.7):int(h*0.9), int(w*0.2):int(w*0.8)]
        if jaw_region.size > 0:
            jaw_edges = cv2.Canny(jaw_region, 50, 150)
            
            jaw_edge_density = np.mean(jaw_edges) / 255.0
            
            jaw_gradient_y = cv2.Sobel(jaw_region, cv2.CV_64F, 0, 1, ksize=3)
            jaw_gradient_strength = np.mean(np.abs(jaw_gradient_y))
            
            jaw_gradient_x = cv2.Sobel(jaw_region, cv2.CV_64F, 1, 0, ksize=3)
            jaw_gradient_x_strength = np.mean(np.abs(jaw_gradient_x))
            
            jaw_feature = (jaw_edge_density * 0.3) + (jaw_gradient_strength / 50.0 * 0.4) + (jaw_gradient_x_strength / 50.0 * 0.3)
        else:
            jaw_feature = 0.5
        features['jaw'] = jaw_feature
        
        # 1.7 Phân tích vùng trán
        forehead_region = enhanced[int(h*0.05):int(h*0.2), int(w*0.25):int(w*0.75)]
        if forehead_region.size > 0:
            forehead_texture = cv2.absdiff(forehead_region, cv2.filter2D(forehead_region, -1, cheek_kernel))
            forehead_smoothness = float(1.0 - (np.mean(forehead_texture) / 128.0))
            
            forehead_gradient_y = cv2.Sobel(forehead_region, cv2.CV_64F, 0, 1, ksize=3)
            forehead_gradient_strength = np.mean(np.abs(forehead_gradient_y))
            
            if forehead_gradient_strength > 20:
                forehead_smoothness = min(1.0, forehead_smoothness * 1.05)
        else:
            forehead_smoothness = 0.5
        features['forehead'] = forehead_smoothness
        
        # 1.8 Cấu trúc hộp sọ (skull structure)
        forehead_width = forehead_region.shape[1] if forehead_region.size > 0 else w*0.5
        jaw_width = jaw_region.shape[1] if jaw_region.size > 0 else w*0.6
        skull_ratio = forehead_width / jaw_width if jaw_width > 0 else 1.0
        features['skull_ratio'] = skull_ratio
        
        # 2. PHỤ THUỘC NHIỀU VÀO FACE_ENCODING
        if face_encoding is not None and len(face_encoding) == 128:
            print("Using face encoding for gender detection")
            
            # 2.1 Phân tích thống kê cơ bản
            encoding_mean = np.mean(face_encoding)
            encoding_std = np.std(face_encoding)
            encoding_max = np.max(face_encoding)
            encoding_min = np.min(face_encoding)
            encoding_q25 = np.percentile(face_encoding, 25)
            encoding_q75 = np.percentile(face_encoding, 75)
            encoding_iqr = encoding_q75 - encoding_q25
            
            features['encoding_mean'] = encoding_mean
            features['encoding_std'] = encoding_std
            features['encoding_range'] = encoding_max - encoding_min
            features['encoding_iqr'] = encoding_iqr
            
            # 2.2 Phân tích các vùng quan trọng của khuôn mặt dựa trên encoding
            # Thông thường, encodings được tạo ra dựa trên grid 4x4 trên khuôn mặt
            # Dựa vào đặc điểm này, ta có thể chia 128 chiều thành 16 vùng, mỗi vùng 8 chiều
            
            # Vùng mắt (thường nằm ở phần 1/3 trên của khuôn mặt, vùng 2-3 trong grid 4x4)
            # Giả định vùng mắt nằm ở indices 16-32 (phần 2 của grid 4x4)
            eye_encoding = face_encoding[16:32]
            features['eye_encoding_mean'] = np.mean(eye_encoding)
            features['eye_encoding_std'] = np.std(eye_encoding)
            
            # Vùng hàm (thường nằm ở phần dưới của khuôn mặt, vùng 13-16 trong grid 4x4)
            # Giả định vùng hàm nằm ở indices 96-128 (phần 13-16 của grid 4x4)
            jaw_encoding = face_encoding[96:128]
            features['jaw_encoding_mean'] = np.mean(jaw_encoding)
            features['jaw_encoding_std'] = np.std(jaw_encoding)
            
            # Vùng mũi (thường nằm ở trung tâm khuôn mặt, vùng 6-7 và 10-11 trong grid 4x4)
            # Giả định vùng mũi nằm ở indices 48-64 (phần 6-7 của grid 4x4)
            nose_encoding = face_encoding[48:64]
            features['nose_encoding_mean'] = np.mean(nose_encoding)
            
            # Vùng miệng (thường nằm ở phần dưới trung tâm, vùng 11-12 trong grid 4x4)
            # Giả định vùng miệng nằm ở indices 80-96 (phần 11-12 của grid 4x4)
            mouth_encoding = face_encoding[80:96]
            features['mouth_encoding_mean'] = np.mean(mouth_encoding)
            
            # 2.3 Phân tích vector đặc trưng
            # Các giá trị cao nhất có thể cho biết những đặc trưng quan trọng nhất
            top_indices = np.argsort(np.abs(face_encoding))[-10:]  # 10 chỉ số với giá trị tuyệt đối lớn nhất
            top_values = face_encoding[top_indices]
            
            # Đếm số lượng giá trị dương và âm trong các đặc trưng quan trọng nhất
            pos_features = np.sum(top_values > 0)
            neg_features = np.sum(top_values < 0)
            
            # Tỉ lệ dương/âm trong các đặc trưng quan trọng nhất có thể là dấu hiệu phân biệt giới tính
            features['top_features_pos_ratio'] = float(pos_features) / max(1, len(top_indices))
            
            # 2.4 Giá trị phân biệt giới tính dựa trên encoding
            # Đặc tính cũng có thể nằm ở mẫu của các giá trị hơn là các giá trị tuyệt đối
            positive_count = np.sum(face_encoding > 0)
            negative_count = np.sum(face_encoding < 0)
            zero_count = np.sum(face_encoding == 0)
            
            features['positive_ratio'] = float(positive_count) / len(face_encoding)
            features['negative_ratio'] = float(negative_count) / len(face_encoding)
            
            # 2.5 Tính trung bình các giá trị dương và âm riêng biệt
            positive_mean = np.mean(face_encoding[face_encoding > 0]) if positive_count > 0 else 0
            negative_mean = np.mean(face_encoding[face_encoding < 0]) if negative_count > 0 else 0
            
            features['positive_mean'] = positive_mean
            features['negative_mean'] = negative_mean
            
            # 2.6 Phân tích độ phức tạp của encoding
            # Độ phức tạp cao (nhiều sự thay đổi giữa các chiều liên tiếp) có thể là dấu hiệu phân biệt
            complexity = np.mean(np.abs(np.diff(face_encoding)))
            features['encoding_complexity'] = complexity
        
        # 3. XỬ LÝ VÀ TÍNH ĐIỂM
        
        # 3.1 Chuẩn hóa các đặc trưng về khoảng [0, 1]
        normalized_features = {}
        feature_ranges = {
            'face_ratio': (0.9, 1.1),
            'smoothness': (0.5, 0.9),
            'cheek_smoothness': (0.5, 0.9),
            'cheek_saturation': (0.1, 0.5),
            'eye_feature': (0.4, 1.0),
            'eyebrow_intensity': (10, 50),
            'symmetry': (0.6, 0.9),
            'jaw': (0.05, 0.25),        # Tăng phạm vi cho jaw
            'forehead': (0.5, 0.9),
            'skull_ratio': (0.8, 1.2),
            # Face encoding ranges - điều chỉnh khoảng để cân bằng
            'encoding_mean': (-0.1, 0.1),
            'encoding_std': (0.2, 0.5),
            'encoding_range': (0.5, 1.5),
            'encoding_iqr': (0.3, 0.8),
            'eye_encoding_mean': (-0.2, 0.2),
            'eye_encoding_std': (0.2, 0.4),
            'jaw_encoding_mean': (-0.2, 0.2),
            'jaw_encoding_std': (0.2, 0.4),
            'nose_encoding_mean': (-0.2, 0.2),
            'mouth_encoding_mean': (-0.2, 0.2),
            'top_features_pos_ratio': (0.3, 0.7),
            'positive_ratio': (0.3, 0.7),
            'negative_ratio': (0.3, 0.7),
            'positive_mean': (0.1, 0.5),
            'negative_mean': (-0.5, -0.1),
            'encoding_complexity': (0.05, 0.2)
        }
        
        for feature_name, feature_value in features.items():
            if feature_name in feature_ranges:
                min_val, max_val = feature_ranges[feature_name]
                # Chuẩn hóa tuyến tính
                norm_value = (feature_value - min_val) / (max_val - min_val)
                # Clip về khoảng [0, 1]
                normalized_features[feature_name] = max(0, min(1, norm_value))
            else:
                # Đối với các đặc trưng không có khoảng đã định nghĩa, đơn giản là clip về [0, 1]
                normalized_features[feature_name] = max(0, min(1, feature_value))
        
        # 3.2 Định nghĩa hệ số cho từng đặc trưng (dương = ủng hộ Woman, âm = ủng hộ Man)
        feature_coefficients = {
            # Hệ số cho đặc trưng hình ảnh
            'face_ratio': -0.5,            # Tăng lên từ -0.4, tỷ lệ mặt cao = nam
            'smoothness': 0.6,             # Giảm từ 0.8 xuống 0.6, da mịn = nữ
            'cheek_smoothness': 0.6,       # Giảm từ 0.7 xuống 0.6, má mịn = nữ
            'cheek_saturation': 0.4,       # Giảm từ 0.6 xuống 0.4, má hồng = nữ
            'eye_feature': 0.5,            # Giảm từ 0.7 xuống 0.5, mắt rõ nét = nữ
            'eyebrow_intensity': -0.8,     # Tăng từ -0.6 lên -0.8, lông mày đậm = nam
            'symmetry': 0.4,               # Giảm từ 0.5 xuống 0.4, đối xứng cao = nữ
            'jaw': -1.0,                   # Tăng từ -0.8 lên -1.0, hàm rõ nét = nam
            'forehead': 0.4,               # Giảm từ 0.5 xuống 0.4, trán mịn = nữ
            'skull_ratio': 0.5,            # Giảm từ 0.6 xuống 0.5, trán rộng so với hàm = nữ
            
            # Hệ số cho đặc trưng face encoding
            'encoding_mean': 0.2,          # Giảm từ 0.3 xuống 0.2
            'encoding_std': 0.4,           # Giảm từ 0.6 xuống 0.4
            'encoding_range': -0.6,        # Tăng từ -0.4 lên -0.6
            'encoding_iqr': 0.3,           # Giảm từ 0.5 xuống 0.3
            'eye_encoding_mean': 0.5,      # Giảm từ 0.7 xuống 0.5
            'eye_encoding_std': 0.4,       # Giảm từ 0.6 xuống 0.4
            'jaw_encoding_mean': -1.0,     # Tăng từ -0.8 lên -1.0
            'jaw_encoding_std': -0.7,      # Tăng từ -0.5 lên -0.7
            'nose_encoding_mean': 0.3,     # Giảm từ 0.4 xuống 0.3
            'mouth_encoding_mean': 0.4,    # Giảm từ 0.6 xuống 0.4
            'top_features_pos_ratio': 0.4, # Giảm từ 0.6 xuống 0.4
            'positive_ratio': 0.3,         # Giảm từ 0.5 xuống 0.3
            'negative_ratio': -0.7,        # Tăng từ -0.5 lên -0.7
            'positive_mean': 0.5,          # Giảm từ 0.7 xuống 0.5
            'negative_mean': -0.9,         # Tăng từ -0.7 lên -0.9
            'encoding_complexity': 0.3     # Giảm từ 0.5 xuống 0.3
        }
        
        # 3.3 Tính điểm cho từng giới tính
        woman_score = 0.0
        man_score = 0.0
        
        # Hệ số trọng số cho các loại đặc trưng
        image_features_weight = 0.5 if face_encoding is not None else 1.0  # Tăng từ 0.4 lên 0.5
        encoding_features_weight = 0.5 if face_encoding is not None else 0.0  # Giảm từ 0.6 xuống 0.5
        
        # Đếm số lượng đặc trưng đã sử dụng cho mỗi loại
        image_features_count = 0
        encoding_features_count = 0
        
        # Đếm số đặc trưng hỗ trợ mỗi giới tính
        woman_features_count = 0
        man_features_count = 0
        
        for feature_name, norm_value in normalized_features.items():
            if feature_name in feature_coefficients:
                coefficient = feature_coefficients[feature_name]
                
                # Xác định trọng số dựa vào loại đặc trưng
                if feature_name in ['encoding_mean', 'encoding_std', 'encoding_range', 'encoding_iqr', 
                                    'eye_encoding_mean', 'eye_encoding_std', 'jaw_encoding_mean', 
                                    'jaw_encoding_std', 'nose_encoding_mean', 'mouth_encoding_mean', 
                                    'top_features_pos_ratio', 'positive_ratio', 'negative_ratio', 
                                    'positive_mean', 'negative_mean', 'encoding_complexity']:
                    weight = encoding_features_weight
                    encoding_features_count += 1
                else:
                    weight = image_features_weight
                    image_features_count += 1
                
                # Tính điểm đóng góp cho từng đặc trưng
                centered_norm_value = norm_value - 0.5  # Đưa về khoảng [-0.5, 0.5]
                feature_contribution = centered_norm_value * coefficient * weight
                
                # In ra chi tiết cho debug
                print(f"- {feature_name}: norm={norm_value:.2f}, centered={centered_norm_value:.2f}, coeff={coefficient:.2f}, weight={weight:.2f}, contribution={feature_contribution:.3f}")
                
                # Đóng góp vào điểm số tổng
                if coefficient > 0:  # Đặc trưng ủng hộ nữ
                    woman_score += feature_contribution
                    if centered_norm_value > 0:
                        woman_features_count += 1
                else:  # Đặc trưng ủng hộ nam
                    man_score -= feature_contribution  # Đảo dấu vì hệ số âm
                    if centered_norm_value > 0:
                        man_features_count += 1
        
        # Chuẩn hóa điểm số dựa trên số lượng đặc trưng đã sử dụng
        total_features_count = image_features_count + encoding_features_count
        if total_features_count > 0:
            woman_score = woman_score / total_features_count
            man_score = man_score / total_features_count
        
        # Loại bỏ bias ủng hộ nữ
        # Thay vì thêm bias thì chúng ta sẽ điều chỉnh woman_score để giảm thiên vị
        woman_score = woman_score * 0.9  # Giảm 10% điểm woman để cân bằng
        
        # Debug info
        print(f"- Counts: image_features={image_features_count}, encoding_features={encoding_features_count}")
        print(f"- Support: woman_features={woman_features_count}, man_features={man_features_count}")
        print(f"- Raw scores: Woman={woman_score:.3f}, Man={man_score:.3f}")
        
        # 3.4 Chuyển đổi điểm thô thành xác suất
        # Đưa điểm về thang [0, 1]
        woman_prob = 0.5 + woman_score
        man_prob = 0.5 + man_score
        
        # Chuẩn hóa tổng xác suất = 1
        total_prob = woman_prob + man_prob
        if total_prob > 0:
            woman_prob = woman_prob / total_prob
            man_prob = man_prob / total_prob
        else:
            woman_prob = 0.5
            man_prob = 0.5
        
        print(f"- Probabilities: Woman={woman_prob:.3f}, Man={man_prob:.3f}")
        
        # 3.5 Xác định giới tính và độ tin cậy
        if woman_prob >= man_prob:
            gender = "Woman"
            confidence = woman_prob
        else:
            gender = "Man"
            confidence = man_prob
        
        # Điều chỉnh độ tin cậy dựa trên độ chênh lệch giữa hai xác suất
        confidence_boost = abs(woman_prob - man_prob) * 0.5
        confidence = 0.5 + confidence_boost
        
        # Nếu sử dụng face_encoding, tăng độ tin cậy
        if face_encoding is not None:
            confidence = min(0.98, confidence * 1.1)
        
        # Giảm độ tin cậy nếu có sự cân bằng giữa số lượng đặc trưng ủng hộ mỗi bên
        feature_balance_ratio = min(woman_features_count, man_features_count) / max(woman_features_count, man_features_count) if max(woman_features_count, man_features_count) > 0 else 0
        if feature_balance_ratio > 0.8:  # Khá cân bằng
            confidence = confidence * 0.9
        
        # Đảm bảo độ tin cậy không vượt quá 0.98
        confidence = min(0.98, max(0.6, confidence))
        
        print(f"- Final decision: {gender} (confidence: {confidence:.2f})")
        
        # Lưu vào bộ nhớ đệm
        img_hash = get_image_hash(face_image)
        encoding_hash = ""
        if face_encoding is not None:
            encoding_str = str(face_encoding.tobytes())
            encoding_hash = hashlib.md5(encoding_str.encode()).hexdigest()[:8]
        
        cache_key = (img_hash, encoding_hash)
        _gender_cache[cache_key] = (gender, confidence)
        
        return gender, confidence
        
    except Exception as e:
        print(f"Error in gender detection: {e}")
        import traceback
        traceback.print_exc()
        return "Man", 0.6

def get_gender_vector(face_image, vector_length=16, face_encoding=None):
    """
    Create feature vector for gender from face image using face encoding.
    
    Args:
        face_image: Face image
        vector_length: Length of output vector
        face_encoding: Face encoding vector (128-dim) nếu có, để cải thiện độ chính xác
        
    Returns:
        ndarray: Gender feature vector (16-dimensional)
    """
    gender_vector = [0.0] * vector_length
    
    if face_image is None or face_image.size == 0:
        gender_vector[14] = 1.0  # Man one-hot
        gender_vector[15] = 0.5  # Medium confidence
        return np.array(gender_vector, dtype=float)
    
    try:
        # Đảm bảo hình ảnh có kích thước 224x224
        if face_image.shape[0] != 224 or face_image.shape[1] != 224:
            face_image = cv2.resize(face_image, (224, 224))
        
        # Xác định giới tính bằng phương pháp dựa vào face_encoding
        gender, confidence = detect_gender(face_image, face_encoding=face_encoding)
        confidence = float(confidence)
        
        # Mã hóa giới tính
        gender_mapping = {"Man": 0, "Woman": 1, "Unknown": 2}
        gender_index = gender_mapping.get(gender, 2)
        
        # Chuẩn bị các không gian màu khác nhau
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
        else:
            gray = face_image.copy()
            color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        
        # Tạo vector đặc trưng cho giới tính
        features = [0.0] * 8  # 8 đặc trưng cơ bản
        
        # - Đặc trưng 0: Giới tính dự đoán
        features[0] = float(gender_index)
        
        # - Đặc trưng 1: Độ tin cậy
        features[1] = min(confidence, 1.0)
        
        # - Đặc trưng 2: One-hot cho Woman
        features[2] = 1.0 if gender == "Woman" else 0.0
        
        # - Đặc trưng 3: One-hot cho Man
        features[3] = 1.0 if gender == "Man" else 0.0
        
        # - Đặc trưng 4-7: Trích xuất các đặc trưng từ hình ảnh và face_encoding
        if face_encoding is not None and len(face_encoding) == 128:
            # Phân tích thống kê cơ bản từ face_encoding
            features[4] = (np.mean(face_encoding) + 0.2) / 0.4  # Giá trị trung bình chuẩn hóa [-0.2, 0.2] -> [0, 1]
            features[5] = min(1.0, np.std(face_encoding) / 0.5)  # Độ lệch chuẩn chuẩn hóa
            
            # Sử dụng các vùng quan trọng của face_encoding
            eye_encoding = face_encoding[16:32]       # Vùng mắt
            jaw_encoding = face_encoding[96:128]      # Vùng hàm
            
            features[6] = (np.mean(eye_encoding) + 0.2) / 0.4   # Giá trị trung bình vùng mắt chuẩn hóa
            features[7] = (np.mean(jaw_encoding) + 0.2) / 0.4   # Giá trị trung bình vùng hàm chuẩn hóa
            
            # Tăng cường đặc trưng vùng hàm nếu có giá trị mạnh
            if np.mean(jaw_encoding) < -0.1:
                features[7] = max(features[7] * 1.2, 1.0)  # Tăng giá trị jawline đặc trưng nam
        else:
            # Tính các đặc trưng từ hình ảnh nếu không có face_encoding
            # Tính lượng đặc trưng khuôn mặt và lưu vào features[4]
            face_ratio = float(face_image.shape[0]) / float(face_image.shape[1])
            features[4] = (face_ratio - 0.8) / 0.4  # Chuẩn hóa về khoảng [0, 1]
            
            # Tính độ mịn da và lưu vào features[5]
            if len(face_image.shape) == 3:
                b, g, r = cv2.split(face_image)
                smoothness = 1.0 - (cv2.Laplacian(g, cv2.CV_64F).var() / 1000.0)
                features[5] = max(0.0, min(1.0, smoothness * 0.9))  # Giảm ảnh hưởng của độ mịn (thiên về nữ)
            
            # Tính đặc trưng jaw và lưu vào features[6]
            jaw_region = gray[int(gray.shape[0]*0.7):int(gray.shape[0]*0.9), 
                              int(gray.shape[1]*0.2):int(gray.shape[1]*0.8)]
            if jaw_region.size > 0:
                jaw_edges = cv2.Canny(jaw_region, 50, 150)
                jaw_feature = np.mean(jaw_edges) / 255.0
                features[6] = min(1.0, jaw_feature * 1.2)  # Tăng ảnh hưởng của đường hàm (thiên về nam)
            
            # Tính đặc trưng mắt và lưu vào features[7]
            eye_region = gray[int(gray.shape[0]*0.2):int(gray.shape[0]*0.45), 
                             int(gray.shape[1]*0.15):int(gray.shape[1]*0.85)]
            if eye_region.size > 0:
                eye_feature = np.std(eye_region) / 50.0
                features[7] = min(1.0, eye_feature)
        
        # Cập nhật 8 thành phần đầu tiên của vector
        for i, feature in enumerate(features):
            gender_vector[i] = float(feature)
        
        # One-hot encoding cho phân loại (8-10)
        gender_vector[8 + gender_index] = 1.0
        
        # Đặc trưng bổ sung (11-12)
        gender_vector[11] = float(features[0])  # Giới tính dự đoán
        gender_vector[12] = float(features[1])  # Độ tin cậy
        
        # One-hot encoding cho giới tính (13-14)
        gender_vector[13] = 1.0 if gender == "Woman" else 0.0
        gender_vector[14] = 1.0 if gender == "Man" else 0.0
        
        # Độ tin cậy (15)
        gender_vector[15] = float(confidence)
        
        # Kiểm tra các giá trị NaN và thay thế bằng 0.0
        for i in range(len(gender_vector)):
            if math.isnan(float(gender_vector[i])):
                gender_vector[i] = 0.0
        
    except Exception as e:
        print(f"Error creating gender vector: {e}")
        import traceback
        traceback.print_exc()
        gender_vector[14] = 1.0  # Man one-hot
        gender_vector[15] = 0.5  # Medium confidence
    
    # Đảm bảo tất cả là float
    for i in range(len(gender_vector)):
        gender_vector[i] = float(gender_vector[i])
    
    return np.array(gender_vector, dtype=float)