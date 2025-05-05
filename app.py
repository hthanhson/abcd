from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import os
import shutil
import face_recognition
import numpy as np
import cv2
from werkzeug.utils import secure_filename
from PIL import Image
import pickle
from deepface import DeepFace
import uuid
import sys
import math
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATA_FOLDER'] = 'data_test'
app.config['ORGANIZED_DATA_FOLDER'] = 'data'
app.config['FEATURES_FILE'] = 'features.pkl'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# Ensure organized data folders exist
os.makedirs(app.config['ORGANIZED_DATA_FOLDER'], exist_ok=True)

# Load or create features dictionary
if os.path.exists(app.config['FEATURES_FILE']):
    with open(app.config['FEATURES_FILE'], 'rb') as f:
        features_db = pickle.load(f)
else:
    features_db = {
        'encodings': [],
        'image_paths': [],
        'emotions': [],
        'ages': [],
        'age_groups': [],
        'skin_colors': []
    }

# Các thông số hiệu chỉnh cho ước tính tuổi (sẽ được cập nhật bởi hàm calibrate)
age_calibration = {
    'multiplier': 1.0,  # Hệ số nhân tổng thể
    'offset': 0,        # Độ lệch cộng/trừ
    'child_factor': 1.0,    # Hệ số điều chỉnh cho trẻ em
    'teen_factor': 1.0,     # Hệ số điều chỉnh cho thanh thiếu niên
    'adult_factor': 1.0,    # Hệ số điều chỉnh cho người lớn
    'senior_factor': 1.0,   # Hệ số điều chỉnh cho người cao tuổi
    'calibrated': False     # Cờ đánh dấu đã hiệu chỉnh chưa
}

def categorize_age(age):
    """Categorize age into groups"""
    if age < 13:
        return "child"
    elif age < 20:
        return "teen"
    elif age < 60:
        return "adult"
    else:
        return "senior"

def classify_skin_color(hsv_face):
    """Classify skin color into White, Yellow, or Black based on HSV values"""
    try:
        # Kiểm tra hình ảnh đầu vào
        if hsv_face is None or hsv_face.size == 0:
            return "unknown"
        
        # Đảm bảo định dạng HSV
        if len(hsv_face.shape) < 3 or hsv_face.shape[2] != 3:
            return "unknown"
        
        # Áp dụng bộ lọc Gaussian để làm mịn trước khi phân tích
        smoothed_face = cv2.GaussianBlur(hsv_face, (5, 5), 0)
        
        # Mở rộng phạm vi màu da để xử lý các trường hợp đặc biệt
        lower_skin = np.array([0, 15, 50], dtype=np.uint8)
        upper_skin = np.array([50, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(smoothed_face, lower_skin, upper_skin)
        
        # Áp dụng các phép toán hình thái học để cải thiện mặt nạ
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Trích xuất pixel màu da
        skin_pixels = smoothed_face[mask > 0]
        
        # Nếu không có đủ pixel màu da, thử lại với ngưỡng thấp hơn
        if skin_pixels.size < 100:
            lower_skin = np.array([0, 10, 40], dtype=np.uint8)
            upper_skin = np.array([60, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(smoothed_face, lower_skin, upper_skin)
            skin_pixels = smoothed_face[mask > 0]
        
        # Nếu vẫn không đủ, sử dụng toàn bộ ảnh và ước tính xác suất
        if skin_pixels.size < 100:
            # Sử dụng các pixel trung tâm của ảnh (giả định đó là vùng khuôn mặt)
            center_region = hsv_face[hsv_face.shape[0]//4:3*hsv_face.shape[0]//4, 
                                  hsv_face.shape[1]//4:3*hsv_face.shape[1]//4]
            if center_region.size > 0:
                skin_pixels = center_region.reshape(-1, 3)
            else:
                skin_pixels = hsv_face.reshape(-1, 3)
        
        # Tính toán giá trị trung bình của các kênh màu HSV
        avg_hue = np.mean(skin_pixels[:, 0])
        avg_saturation = np.mean(skin_pixels[:, 1])
        avg_value = np.mean(skin_pixels[:, 2])
        
        # Tính độ lệch chuẩn để đánh giá độ tin cậy
        std_hue = np.std(skin_pixels[:, 0])
        std_saturation = np.std(skin_pixels[:, 1])
        std_value = np.std(skin_pixels[:, 2])
        
        # Đánh giá độ tin cậy dựa trên độ lệch chuẩn
        reliability = 1.0
        if std_hue > 20 or std_saturation > 60 or std_value > 60:
            reliability = 0.7  # Giảm độ tin cậy nếu có sự biến động lớn
        
        # Phân loại dựa trên các ngưỡng điều chỉnh
        # White skin: Giá trị cao, độ bão hòa thấp
        if avg_value > 170 and avg_saturation < 90:
            return "White"
            
        # Black skin: Giá trị thấp
        elif avg_value < 130:
            return "Black"
            
        # Yellow skin: Tông màu vàng, giá trị trung bình, độ bão hòa trung bình
        elif 5 <= avg_hue <= 30 and avg_saturation > 30:
            return "Yellow"
            
        # Các trường hợp còn lại
        else:
            # Sử dụng các quy tắc dự phòng dựa trên tỷ lệ và độ tin cậy
            if avg_value > 150:
                if avg_hue > 10 and avg_hue < 25 and avg_saturation > 40:
                    return "Yellow"
                else:
                    return "White"
            elif avg_value < 110:
                return "Black"
            else:
                # Xác định màu da dựa trên sự kết hợp của các yếu tố
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

def extract_age_from_filename(filename):
    """Extract actual age from filename if available (for calibration)"""
    # Giả sử tên file có định dạng như: 35_1_0_20170109205304456.jpg
    # Trong đó 35 là tuổi thật
    try:
        parts = filename.split('_')
        if len(parts) >= 1 and parts[0].isdigit():
            return int(parts[0])
    except:
        pass
    return None

def calibrate_age_estimation():
    """Calibrate age estimation based on files with known ages in their names"""
    global age_calibration
    
    data_folder = app.config['DATA_FOLDER']
    image_files = [f for f in os.listdir(data_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Thu thập các cặp tuổi thật và tuổi ước tính
    age_pairs = []
    age_errors_by_group = {
        'child': [],
        'teen': [],
        'adult': [],
        'senior': []
    }
    
    for img_file in image_files[:100]:  # Giới hạn ở 100 ảnh đầu để tối ưu thời gian
        # Cố gắng trích xuất tuổi thật từ tên file
        actual_age = extract_age_from_filename(img_file)
        if actual_age is None:
            continue
        
        # Trích xuất đặc trưng và ước tính tuổi
        img_path = os.path.join(data_folder, img_file)
        try:
            image = face_recognition.load_image_file(img_path)
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                continue
                
            # Sử dụng DeepFace để ước tính tuổi
            analysis = DeepFace.analyze(img_path, actions=['age'], enforce_detection=False)
            raw_age = analysis[0]['age']
            
            # Trích xuất vùng khuôn mặt
            top, right, bottom, left = face_locations[0]
            face_image = image[top:bottom, left:right]
            
            # Ước tính tuổi ban đầu (chưa hiệu chỉnh tổng thể)
            estimated_age = adjust_age_estimation(raw_age, face_image)
            
            # Lưu cặp tuổi để phân tích
            age_pairs.append((actual_age, estimated_age))
            
            # Phân loại theo nhóm tuổi
            age_group = categorize_age(actual_age)
            age_errors_by_group[age_group].append(actual_age - estimated_age)
        except Exception as e:
            print(f"Error calibrating with {img_file}: {e}")
    
    # Nếu không đủ dữ liệu, không hiệu chỉnh
    if len(age_pairs) < 10:
        print("Not enough data for age calibration")
        return False
    
    # Tính toán hệ số hiệu chỉnh tổng thể bằng hồi quy tuyến tính
    actuals = np.array([pair[0] for pair in age_pairs])
    estimates = np.array([pair[1] for pair in age_pairs])
    
    # Tính hệ số nhân và độ lệch tốt nhất
    if len(actuals) > 1:
        slope, intercept = np.polyfit(estimates, actuals, 1)
        age_calibration['multiplier'] = slope
        age_calibration['offset'] = intercept
    
    # Tính hệ số hiệu chỉnh cho từng nhóm tuổi
    for group, errors in age_errors_by_group.items():
        if len(errors) > 5:  # Cần ít nhất 5 mẫu cho nhóm tuổi
            avg_error = np.mean(errors)
            if group == 'child':
                age_calibration['child_factor'] = 1.0 + avg_error / 10
            elif group == 'teen':
                age_calibration['teen_factor'] = 1.0 + avg_error / 20
            elif group == 'adult':
                age_calibration['adult_factor'] = 1.0 + avg_error / 40
            elif group == 'senior':
                age_calibration['senior_factor'] = 1.0 + avg_error / 50
    
    # Đánh dấu đã hiệu chỉnh
    age_calibration['calibrated'] = True
    print(f"Age calibration completed with {len(age_pairs)} samples")
    print(f"Calibration factors: {age_calibration}")
    
    return True

def adjust_age_estimation(age, face_image):
    """Improved age estimation using multiple facial features and advanced image analysis"""
    # Convert age to float for precise calculations
    age = float(age)
    
    # Kiểm tra tuổi ban đầu từ DeepFace
    if age <= 0:
        age = 25  # Giá trị mặc định nếu DeepFace trả về tuổi không hợp lệ (tăng từ 15 lên 25)
    
    # In thông tin để debug
    print(f"Initial DeepFace age: {age}")
    
    try:
        # Get face dimensions
        height, width = face_image.shape[:2]
        face_size_ratio = width / height
        
        # 1. Detailed facial proportions analysis
        # Convert to grayscale for feature detection
        gray_face = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding to enhance features
        thresh = cv2.adaptiveThreshold(gray_face, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
        
        # 2. Advanced wrinkle and texture analysis
        # Horizontal edge detection (for wrinkles)
        sobelx = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        sobel_mean = np.mean(abs_sobelx)
        
        # Vertical edge detection (for facial structure)
        sobely = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobely = np.absolute(sobely)
        sobely_mean = np.mean(abs_sobely)
        
        # Texture variance (skin smoothness)
        texture_variance = np.var(gray_face)
        
        # Calculate wrinkle density for forehead, eye area, and mouth area
        forehead_region = gray_face[0:int(height/3), :]
        eye_region = gray_face[int(height/3):int(2*height/3), :]
        mouth_region = gray_face[int(2*height/3):, :]
        
        forehead_edges = cv2.Sobel(forehead_region, cv2.CV_64F, 1, 0, ksize=3)
        eye_edges = cv2.Sobel(eye_region, cv2.CV_64F, 1, 0, ksize=3)
        mouth_edges = cv2.Sobel(mouth_region, cv2.CV_64F, 1, 0, ksize=3)
        
        forehead_wrinkle_score = np.mean(np.absolute(forehead_edges))
        eye_wrinkle_score = np.mean(np.absolute(eye_edges))
        mouth_wrinkle_score = np.mean(np.absolute(mouth_edges))
        
        # 3. Color variance analysis (skin tone evenness decreases with age)
        hsv_face = cv2.cvtColor(face_image, cv2.COLOR_RGB2HSV)
        skin_tone_variance = np.var(hsv_face[:,:,1])  # Variance in saturation channel
        
        # 4. Apply age corrections based on detailed analysis với hệ số nhỏ hơn
        # Lưu trữ tuổi ban đầu để phục vụ so sánh
        original_age = age
        
        # 4.1 Face shape correction (more refined) - HẠN CHẾ ẢNH HƯỞNG của hình dạng khuôn mặt
        if face_size_ratio > 0.95:  # Rounder face
            if age > 30:
                # Rounder faces appear younger in adults - giảm tác động
                age_adjustment = -0.02 * (age - 30)  # Giảm tác động
                age = age + age_adjustment
        elif face_size_ratio < 0.85:  # More elongated face
            if age < 40:
                # Elongated faces appear older in younger people - giảm tác động
                age_adjustment = 0.02 * (40 - age)  # Giảm tác động
                age = age + age_adjustment
        
        # 4.2 Wrinkle-based corrections - GIẢM MẠNH MỨC ĐỘ ẢNH HƯỞNG
        # Forehead wrinkles are significant for age
        if forehead_wrinkle_score > 18:  # Tăng ngưỡng từ 15 lên 18
            if age < 45:
                age += (forehead_wrinkle_score - 18) * 0.08  # Giảm tác động
        elif forehead_wrinkle_score < 8 and age > 30:
            age -= (8 - forehead_wrinkle_score) * 0.03  # Giảm tác động
        
        # Eye wrinkles (crow's feet)
        if eye_wrinkle_score > 20:  # Tăng ngưỡng từ 18 lên 20
            if age < 50:
                age += (eye_wrinkle_score - 20) * 0.06  # Giảm tác động
        
        # Mouth wrinkles (nasolabial folds)
        if mouth_wrinkle_score > 22:  # Tăng ngưỡng từ 20 lên 22
            if age < 55:
                age += (mouth_wrinkle_score - 22) * 0.07  # Giảm tác động
        
        # 4.3 Texture variance indicates skin smoothness - GIẢM MỨC ĐỘ ẢNH HƯỞNG
        if texture_variance < 800 and age > 40:
            # Smoother skin for reported age, reduce age
            age_adjustment = min(2, (800 - texture_variance) * 0.003)  # Giảm tác động
            age -= age_adjustment
        elif texture_variance > 1500 and age < 50:
            # More textured skin for reported age, increase age
            age_adjustment = min(3, (texture_variance - 1500) * 0.003)  # Giảm tác động
            age += age_adjustment
        
        # 4.4 Skin tone evenness - GIẢM MỨC ĐỘ ẢNH HƯỞNG
        if skin_tone_variance < 100 and age > 35:
            # Even skin tone suggests younger appearance
            age -= min(1.5, (100 - skin_tone_variance) * 0.008)  # Giảm tác động
        elif skin_tone_variance > 200 and age < 45:
            # Uneven skin tone suggests older appearance
            age += min(2, (skin_tone_variance - 200) * 0.004)  # Giảm tác động
        
        # Kiểm tra xem tuổi có thay đổi quá nhiều từ ước tính ban đầu không
        if abs(age - original_age) > original_age * 0.4:  # Nếu thay đổi >40%
            # Điều chỉnh lại bằng trung bình có trọng số
            age = original_age * 0.6 + age * 0.4  # Nghiêng về tuổi ban đầu hơn
        
        # 5. Improved confidence-based regression toward mean
        # Apply different regression models for different age ranges
        if age < 10:
            # Children
            regression_strength = 0.1  # Giảm từ 0.15
            mean_age = 8
            age = age * (1 - regression_strength) + mean_age * regression_strength
        elif age < 20:
            # Teens
            regression_strength = 0.08  # Giảm từ 0.1
            mean_age = 16
            age = age * (1 - regression_strength) + mean_age * regression_strength
        elif age > 75:
            # Elderly
            regression_strength = 0.08  # Giảm từ 0.1
            mean_age = 80
            age = age * (1 - regression_strength) + mean_age * regression_strength
        
        # 6. Apply calibration factors nếu có
        if age_calibration['calibrated']:
            # Hạn chế mức độ hiệu chỉnh của calibration
            calibrated_age = age * age_calibration['multiplier'] + age_calibration['offset']
            # Chỉ áp dụng 70% hiệu ứng hiệu chỉnh để tránh chệch quá xa
            age = age * 0.3 + calibrated_age * 0.7
            
            # Apply specific group calibration với giới hạn tác động
            age_group = categorize_age(age)
            if age_group == 'child':
                factor = max(0.8, min(1.2, age_calibration['child_factor']))  # Giới hạn factor
                age = age * factor
            elif age_group == 'teen':
                factor = max(0.8, min(1.2, age_calibration['teen_factor']))  # Giới hạn factor
                age = age * factor
            elif age_group == 'adult':
                factor = max(0.9, min(1.1, age_calibration['adult_factor']))  # Giới hạn factor
                age = age * factor
            elif age_group == 'senior':
                factor = max(0.9, min(1.1, age_calibration['senior_factor']))  # Giới hạn factor
                age = age * factor
    
    except Exception as e:
        print(f"Error in age estimation adjustments: {e}")
        # Nếu xảy ra lỗi trong các điều chỉnh, giữ nguyên tuổi ban đầu
        pass
    
    # 7. Final bounds checking and rounding
    # Bảo đảm tuổi luôn trong phạm vi hợp lý
    age = max(1, min(100, age))  # Tuổi tối thiểu là 1, tối đa là 100
    
    # Thêm thuật toán smoothing dựa trên nhóm tuổi
    age_group = categorize_age(age)
    if age_group == 'child':
        # Trẻ em thường có tuổi chẵn hơn (e.g., 3, 4, 5...)
        # Làm tròn đến 0.5 gần nhất rồi làm tròn lên/xuống
        age = round(round(age * 2) / 2)
    elif age_group == 'teen' or age_group == 'adult':
        # Làm tròn đến số nguyên gần nhất
        age = round(age)
    else:  # senior
        # Người lớn tuổi thường làm tròn đến 5 năm gần nhất (e.g., 65, 70, 75)
        # Nhưng chúng ta vẫn giữ độ chính xác cao hơn
        age = round(age)
        
    # In thông tin debug cuối cùng
    print(f"Final adjusted age: {age}")
    
    return age

def improve_emotion_detection(emotion, face_image, emotions_dict):
    """Improve emotion detection by checking consistency and using advanced image analysis"""
    try:
        # Kiểm tra xem face_image có hợp lệ không
        if face_image is None or face_image.size == 0:
            return "neutral"  # Trả về giá trị mặc định nếu ảnh không hợp lệ
            
        # Nâng cao độ tương phản của ảnh để phát hiện đặc trưng tốt hơn
        gray_face = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray_face)
        
        # Kiểm tra độ tin cậy
        max_confidence = max(emotions_dict.values())
        
        # Giá trị phân biệt dựa trên độ sắc nét của ảnh
        # Tính độ sắc nét bằng phương sai Laplacian
        laplacian_var = cv2.Laplacian(enhanced_gray, cv2.CV_64F).var()
        
        # Điều chỉnh ngưỡng tin cậy dựa trên độ nét của ảnh
        confidence_threshold = 0.35
        if laplacian_var < 100:  # Ảnh mờ
            confidence_threshold = 0.30  # Giảm ngưỡng tin cậy cho ảnh mờ
        
        # Nếu độ tin cậy rất thấp, thử với các phương pháp khác
        if max_confidence < confidence_threshold:
            # 1. Phân tích cân đối khuôn mặt
            # Chia đôi khuôn mặt để kiểm tra tính cân đối
            try:
                height, width = face_image.shape[:2]
                left_half = face_image[:, :width//2]
                right_half = face_image[:, width//2:]
                right_half_flipped = cv2.flip(right_half, 1)
                
                # Resize cả hai nửa để có cùng kích thước
                if left_half.shape != right_half_flipped.shape:
                    min_height = min(left_half.shape[0], right_half_flipped.shape[0])
                    min_width = min(left_half.shape[1], right_half_flipped.shape[1])
                    left_half = left_half[:min_height, :min_width]
                    right_half_flipped = right_half_flipped[:min_height, :min_width]
                
                # Tính toán sự khác biệt giữa hai nửa
                diff = np.mean(cv2.absdiff(left_half, right_half_flipped))
                symmetry = 1 - min(1, diff / 50)  # Chuẩn hóa về 0-1
                
                # Điều chỉnh dựa trên tính cân đối
                if symmetry > 0.8:  # Khuôn mặt rất cân đối
                    if emotions_dict.get('neutral', 0) > 0.2:
                        return 'neutral'
                    elif emotions_dict.get('happy', 0) > 0.2:
                        return 'happy'
                elif symmetry < 0.6:  # Khuôn mặt không cân đối
                    if emotions_dict.get('surprise', 0) > 0.15:
                        return 'surprise'
                    if emotions_dict.get('disgust', 0) > 0.15:
                        return 'disgust'
                    if emotions_dict.get('sad', 0) > 0.15:
                        return 'sad'
            except Exception as e:
                print(f"Error analyzing face symmetry: {e}")
            
            # 2. Phân tích đặc trưng của miệng
            try:
                # Phân tích vùng miệng (1/3 dưới của khuôn mặt)
                mouth_region = face_image[2*face_image.shape[0]//3:, :]
                mouth_gray = cv2.cvtColor(mouth_region, cv2.COLOR_RGB2GRAY)
                
                # Tạo binary mask để phát hiện độ mở của miệng
                _, mouth_binary = cv2.threshold(mouth_gray, 60, 255, cv2.THRESH_BINARY)
                
                # Tính phần trăm pixel trắng (để dự đoán miệng mở/đóng)
                white_pixel_percentage = np.sum(mouth_binary) / (mouth_binary.size * 255)
                
                # Nếu miệng mở (tỷ lệ pixel trắng cao), có thể là happy, surprise hoặc fear
                if white_pixel_percentage > 0.3:
                    # Xác định cảm xúc dựa trên tỷ lệ giữa các cảm xúc trong emotions_dict
                    smile_emotions = {e: v for e, v in emotions_dict.items() if e in ['happy', 'surprise', 'fear']}
                    if smile_emotions:
                        return max(smile_emotions.items(), key=lambda x: x[1])[0]
            except Exception as e:
                print(f"Error analyzing mouth region: {e}")
            
            # 3. Kiểm tra vùng mắt
            try:
                # Phân tích vùng mắt (1/3 giữa của khuôn mặt)
                eye_region = face_image[face_image.shape[0]//3:2*face_image.shape[0]//3, :]
                eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_RGB2GRAY)
                
                # Sử dụng Canny edge detector để phát hiện đường viền
                edges = cv2.Canny(eye_gray, 50, 150)
                edge_density = np.sum(edges) / edges.size
                
                # Mật độ cạnh cao trong vùng mắt có thể chỉ ra cảm xúc tiêu cực
                if edge_density > 0.1:
                    negative_emotions = {e: v for e, v in emotions_dict.items() if e in ['anger', 'fear', 'sad']}
                    if negative_emotions:
                        return max(negative_emotions.items(), key=lambda x: x[1])[0]
            except Exception as e:
                print(f"Error analyzing eye region: {e}")
        
        # So sánh giữa hai cảm xúc nhận diện được với độ tin cậy cao nhất
        # Nếu sự khác biệt nhỏ, chọn cảm xúc dễ nhận biết hơn
        emotions_sorted = sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True)
        if len(emotions_sorted) > 1:
            first, second = emotions_sorted[0], emotions_sorted[1]
            
            # Nếu sự khác biệt về độ tin cậy nhỏ
            if first[1] - second[1] < 0.15:
                # Ưu tiên các cảm xúc dễ nhận biết hơn (neutral, happy)
                easy_emotions = ['neutral', 'happy']
                if second[0] in easy_emotions:
                    return second[0]
                
                # Nếu cả hai đều là cảm xúc khó nhận biết, ưu tiên cảm xúc dễ nhìn thấy hơn
                difficult_pairs = [('fear', 'surprise'), ('anger', 'disgust'), ('sad', 'disgust')]
                for pair in difficult_pairs:
                    if first[0] in pair and second[0] in pair:
                        if first[0] == 'surprise' or first[0] == 'anger' or first[0] == 'sad':
                            return first[0]
                        else:
                            return second[0]
        
        # Trả về cảm xúc ban đầu nếu không có điều chỉnh nào được áp dụng
        return emotion
    
    except Exception as e:
        print(f"Error in emotion improvement: {e}")
        return emotion  # Trả về cảm xúc ban đầu nếu có lỗi

def extract_features(image_path):
    """Extract facial features: encoding, emotion, age, and skin color with improved accuracy"""
    try:
        # Load image
        image = face_recognition.load_image_file(image_path)
        
        # Tiền xử lý ảnh để cải thiện chất lượng
        # 1. Chuyển sang grayscale để tăng độ tương phản
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 2. Cải thiện độ tương phản
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray_image)
        
        # 3. Giảm nhiễu
        denoised = cv2.fastNlMeansDenoising(enhanced_gray, None, 10, 7, 21)
        
        # Tìm khuôn mặt bằng nhiều phương pháp
        # Thử với face_recognition trước
        face_locations = face_recognition.face_locations(image)
        
        # Nếu không tìm thấy khuôn mặt, thử với độ chính xác thấp hơn 
        if not face_locations:
            # Thử với model CNN mặc dù chậm hơn nhưng chính xác hơn cho trường hợp khó
            face_locations = face_recognition.face_locations(image, model="cnn")
        
        # Nếu vẫn không tìm thấy, thử với OpenCV cascade 
        if not face_locations:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            cv_faces = face_cascade.detectMultiScale(
                denoised, 
                scaleFactor=1.1, 
                minNeighbors=3, 
                minSize=(50, 50),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Chuyển định dạng từ OpenCV sang face_recognition
            if len(cv_faces) > 0:
                face_locations = []
                for (x, y, w, h) in cv_faces:
                    face_locations.append((y, x + w, y + h, x))  # top, right, bottom, left
        
        # Nếu vẫn không tìm thấy, thử với các tham số khác
        if not face_locations:
            # Thử với tham số MinNeighbors thấp hơn
            cv_faces = face_cascade.detectMultiScale(
                denoised, 
                scaleFactor=1.05, 
                minNeighbors=2, 
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(cv_faces) > 0:
                face_locations = []
                for (x, y, w, h) in cv_faces:
                    face_locations.append((y, x + w, y + h, x))
        
        # Nếu vẫn không tìm thấy mặt, thử phân tích cả ảnh (không lý tưởng nhưng tốt hơn là không có gì)
        if not face_locations:
            # Giả định một khuôn mặt ở chính giữa ảnh
            height, width = image.shape[:2]
            center_x, center_y = width // 2, height // 2
            face_size = min(width, height) // 2
            
            # Tạo một khuôn mặt ảo
            top = max(0, center_y - face_size)
            right = min(width, center_x + face_size)
            bottom = min(height, center_y + face_size)
            left = max(0, center_x - face_size)
            
            face_locations = [(top, right, bottom, left)]
            print(f"No face detected in {image_path}, using center of image")
            
        # Bây giờ chúng ta có face_locations, tiếp tục với phân tích
        
        # Lấy encoding cho khuôn mặt
        face_encodings = face_recognition.face_encodings(image, face_locations)
        if not face_encodings and len(face_locations) > 0:
            # Mở rộng khu vực khuôn mặt nếu không lấy được encoding
            expanded_face_locations = []
            for (top, right, bottom, left) in face_locations:
                height, width = image.shape[:2]
                # Mở rộng 20% mỗi hướng
                expand_px = min(bottom - top, right - left) // 5
                new_top = max(0, top - expand_px)
                new_right = min(width, right + expand_px)
                new_bottom = min(height, bottom + expand_px)
                new_left = max(0, left - expand_px)
                expanded_face_locations.append((new_top, new_right, new_bottom, new_left))
                
            # Thử lấy encoding với khu vực mở rộng
            face_encodings = face_recognition.face_encodings(image, expanded_face_locations)
            if face_encodings:
                # Nếu thành công, cập nhật face_locations
                face_locations = expanded_face_locations
        
        # Nếu vẫn không lấy được encoding
        if not face_encodings:
            # Đối với trường hợp khó, chúng ta sẽ tạo một encoding giả
            # Encoding rỗng với các giá trị trung bình
            dummy_encoding = np.zeros(128)
            # Tạo một số biến thể ngẫu nhiên để không giống hệt nhau
            random_factor = np.random.normal(0, 0.01, 128)
            dummy_encoding += random_factor
            dummy_encoding /= np.linalg.norm(dummy_encoding)  # Chuẩn hóa
            print(f"Could not extract encoding from {image_path}, using fallback")
            encoding = dummy_encoding
        else:
            encoding = face_encodings[0]  # Sử dụng khuôn mặt đầu tiên
        
        # Trích xuất vùng khuôn mặt cho xử lý tiếp theo
        top, right, bottom, left = face_locations[0]
        face_image = image[top:bottom, left:right]
        
        # Phân tích cảm xúc và tuổi bằng DeepFace
        try:
            # Sử dụng thử cả hình ảnh và vùng mặt đã trích xuất
            try:
                analysis = DeepFace.analyze(
                    face_image, 
                    actions=['emotion', 'age'], 
                    enforce_detection=False,
                    detector_backend='retinaface'
                )
            except:
                # Nếu phân tích trực tiếp khuôn mặt không thành công, thử với ảnh gốc
                analysis = DeepFace.analyze(
                    image_path, 
                    actions=['emotion', 'age'], 
                    enforce_detection=False,
                    detector_backend='opencv'  # Thử với opencv nếu retinaface thất bại
                )
            
            # Đảm bảo định dạng nhất quán
            if isinstance(analysis, list):
                analysis = analysis[0]
            
            # Lấy dữ liệu cảm xúc và cải thiện độ chính xác
            emotions_dict = analysis['emotion']
            emotion = analysis['dominant_emotion']
            improved_emotion = improve_emotion_detection(emotion, face_image, emotions_dict)
            
            # Lấy tuổi và điều chỉnh
            raw_age = analysis['age']
            adjusted_age = adjust_age_estimation(raw_age, face_image)
            age_group = categorize_age(adjusted_age)
            
            # Phân tích màu da
            hsv_face = cv2.cvtColor(face_image, cv2.COLOR_RGB2HSV)
            skin_color_category = classify_skin_color(hsv_face)
            
            return encoding, improved_emotion, adjusted_age, age_group, skin_color_category
            
        except Exception as e:
            print(f"Error with DeepFace analysis: {e}")
            # Trả về giá trị mặc định nếu không phân tích được
            return encoding, "neutral", 30, "adult", "unknown"
            
    except Exception as e:
        print(f"Critical error analyzing image {image_path}: {e}")
        return None, None, None, None, None

def copy_image_to_category_folders(img_path, emotion, age_group, skin_color):
    """Copy image to appropriate category folders in the organized data directory"""
    try:
        filename = os.path.basename(img_path)
        
        # Copy to emotion folder
        emotion_dir = os.path.join(app.config['ORGANIZED_DATA_FOLDER'], 'emotions', emotion)
        os.makedirs(emotion_dir, exist_ok=True)
        shutil.copy(img_path, os.path.join(emotion_dir, filename))
        
        # Copy to age group folder
        age_dir = os.path.join(app.config['ORGANIZED_DATA_FOLDER'], 'age', age_group)
        os.makedirs(age_dir, exist_ok=True)
        shutil.copy(img_path, os.path.join(age_dir, filename))
        
        # Copy to skin color folder
        if skin_color != "unknown":
            skin_dir = os.path.join(app.config['ORGANIZED_DATA_FOLDER'], 'skin', skin_color)
            os.makedirs(skin_dir, exist_ok=True)
            shutil.copy(img_path, os.path.join(skin_dir, filename))
        
        return True
    except Exception as e:
        print(f"Error copying image to category folders: {e}")
        return False

def build_database():
    """Process all images in data folder and build feature database"""
    global features_db
    
    # Hiệu chuẩn ước tính tuổi nếu có thể
    print("Calibrating age estimation...")
    calibrate_age_estimation()
    
    data_folder = app.config['DATA_FOLDER']
    image_files = [f for f in os.listdir(data_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    features_db = {
        'encodings': [],
        'image_paths': [],
        'emotions': [],
        'ages': [],
        'age_groups': [],
        'skin_colors': []
    }
    
    total = len(image_files)
    print(f"Processing {total} images...")
    
    for i, img_file in enumerate(image_files):
        print(f"Processing image {i+1}/{total}: {img_file}")
        img_path = os.path.join(data_folder, img_file)
        encoding, emotion, age, age_group, skin_color = extract_features(img_path)
        
        if encoding is not None:
            features_db['encodings'].append(encoding)
            features_db['image_paths'].append(img_path)
            features_db['emotions'].append(emotion)
            features_db['ages'].append(age)
            features_db['age_groups'].append(age_group)
            features_db['skin_colors'].append(skin_color)
            
            # Organize images by categories
            copy_image_to_category_folders(img_path, emotion, age_group, skin_color)
    
    # Save the features database
    with open(app.config['FEATURES_FILE'], 'wb') as f:
        pickle.dump(features_db, f)
    
    print(f"Database built with {len(features_db['encodings'])} faces")
    return len(features_db['encodings'])

def find_similar_faces(query_encoding, top_n=3):
    """Find top N similar faces based on facial encoding"""
    if not features_db['encodings']:
        return []
    
    # Calculate face distances
    face_distances = face_recognition.face_distance(features_db['encodings'], query_encoding)
    
    # Sort and get top N matches
    indices = np.argsort(face_distances)[:top_n]
    
    # Result list
    result = []
    for i in indices:
        similarity = 1 - face_distances[i]  # Convert distance to similarity (0-1)
        result.append({
            'image_path': features_db['image_paths'][i],
            'similarity': float(similarity),
            'emotion': features_db['emotions'][i],
            'age': features_db['ages'][i],
            'age_group': features_db['age_groups'][i],
            'skin_color': features_db['skin_colors'][i]
        })
    
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/build-database', methods=['POST'])
def build_db_route():
    count = build_database()
    return jsonify({
        'status': 'success',
        'message': f'Database built with {count} faces'
    })

@app.route('/search', methods=['POST'])
def search():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})
    
    # Save the uploaded file
    filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    try:
        # Extract features
        encoding, emotion, age, age_group, skin_color = extract_features(file_path)
        
        if encoding is None:
            os.remove(file_path)  # Clean up
            return jsonify({
                'status': 'error',
                'message': 'No face detected in the uploaded image'
            })
        
        # Find similar faces
        similar_faces = find_similar_faces(encoding)
        
        # Prepare response with base64 encoded images
        # Read the query image and convert to base64
        with open(file_path, "rb") as image_file:
            query_image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Process similar faces to include base64 images
        results = []
        for face in similar_faces:
            # Read and convert similar face image to base64
            with open(face['image_path'], "rb") as image_file:
                face_image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Add to results list
            results.append({
                'image': face_image_data,
                'similarity': face['similarity'],
                'features': {
                    'emotion': face['emotion'],
                    'age': face['age'],
                    'age_group': face['age_group'],
                    'skin_color': face['skin_color'],
                    'gender': 'unknown'  # Add placeholder for compatibility
                }
            })
        
        # Return response
        return jsonify({
            'status': 'success',
            'query_image': query_image_data,
            'query_features': {
                'emotion': emotion,
                'age': age,
                'age_group': age_group,
                'skin_color': skin_color,
                'gender': 'unknown'  # Add placeholder for compatibility
            },
            'results': results
        })
    except Exception as e:
        print(f"Error in search: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error processing image: {str(e)}'
        })

@app.route('/filter', methods=['POST'])
def filter_images():
    emotion = request.form.get('emotion', '')
    min_age = int(request.form.get('min_age', 0))
    max_age = int(request.form.get('max_age', 100))
    age_group = request.form.get('age_group', '')
    skin_color = request.form.get('skin_color', '')
    
    results = []
    
    for i, img_path in enumerate(features_db['image_paths']):
        matches_emotion = not emotion or features_db['emotions'][i] == emotion
        matches_age = min_age <= features_db['ages'][i] <= max_age
        matches_age_group = not age_group or features_db['age_groups'][i] == age_group
        matches_skin_color = not skin_color or features_db['skin_colors'][i] == skin_color
        
        if matches_emotion and matches_age and matches_age_group and matches_skin_color:
            results.append({
                'image_path': img_path,
                'emotion': features_db['emotions'][i],
                'age': features_db['ages'][i],
                'age_group': features_db['age_groups'][i],
                'skin_color': features_db['skin_colors'][i]
            })
    
    return jsonify({
        'status': 'success',
        'results': results
    })

@app.route('/api/category/<category_type>/<category_name>', methods=['GET'])
def get_category_images(category_type, category_name):
    """Return a list of images from the specified category folder"""
    if category_type not in ['emotions', 'age']:
        return jsonify({
            'status': 'error',
            'message': 'Invalid category type'
        })
    
    category_path = os.path.join(app.config['ORGANIZED_DATA_FOLDER'], category_type, category_name)
    if not os.path.exists(category_path):
        return jsonify({
            'status': 'error',
            'message': f'Category folder {category_type}/{category_name} does not exist'
        })
    
    # Get all image files from the category
    image_files = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Create full paths and extract feature information
    images = []
    for img_file in image_files:
        img_path = os.path.join(category_path, img_file)
        relative_path = os.path.join(category_type, category_name, img_file)
        
        # Look up the image in the features database
        original_path = None
        for i, path in enumerate(features_db['image_paths']):
            if os.path.basename(path) == img_file:
                original_path = path
                images.append({
                    'image_path': os.path.join('data', relative_path),
                    'original_path': path,
                    'emotion': features_db['emotions'][i],
                    'age': features_db['ages'][i],
                    'age_group': features_db['age_groups'][i],
                    'skin_color': features_db['skin_colors'][i]
                })
                break
        
        # If not found in database, add with minimal info
        if not original_path:
            images.append({
                'image_path': os.path.join('data', relative_path),
                'original_path': None,
                'emotion': category_name if category_type == 'emotions' else 'unknown',
                'age': 0,
                'age_group': category_name if category_type == 'age' else 'unknown',
                'skin_color': 0
            })
    
    return jsonify({
        'status': 'success',
        'category_type': category_type,
        'category_name': category_name,
        'count': len(images),
        'images': images
    })

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# Thêm route mới cho thư mục data được tổ chức
@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory(app.config['ORGANIZED_DATA_FOLDER'], filename)

# Thêm các route cụ thể cho thư mục uploads và data_test
@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/data_test/<path:filename>')
def serve_data_test(filename):
    return send_from_directory(app.config['DATA_FOLDER'], filename)

# Giữ lại route gốc để tương thích ngược
@app.route('/images/<path:filename>')
def get_image(filename):
    # Xác định thư mục gốc dựa trên đường dẫn
    if filename.startswith(app.config['UPLOAD_FOLDER']):
        return send_from_directory(
            app.config['UPLOAD_FOLDER'], 
            filename.replace(app.config['UPLOAD_FOLDER'] + '/', '')
        )
    elif filename.startswith(app.config['DATA_FOLDER']):
        return send_from_directory(
            app.config['DATA_FOLDER'], 
            filename.replace(app.config['DATA_FOLDER'] + '/', '')
        )
    elif filename.startswith(app.config['ORGANIZED_DATA_FOLDER']):
        return send_from_directory(
            app.config['ORGANIZED_DATA_FOLDER'],
            filename.replace(app.config['ORGANIZED_DATA_FOLDER'] + '/', '')
        )
    return send_from_directory(os.path.dirname(filename), os.path.basename(filename))

if __name__ == '__main__':
    # Check if we should just build the database
    if len(sys.argv) > 1 and sys.argv[1] == 'build_db':
        build_database()
    else:
        app.run(debug=True) 