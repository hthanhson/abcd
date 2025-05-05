#!/usr/bin/env python3
"""
Benchmark script for the Face Recognition System
This script evaluates the performance of the face recognition system.
"""

import os
import time
import pickle
import face_recognition
import numpy as np
from deepface import DeepFace
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random
import cv2
import math
import argparse
from datetime import datetime

# Configuration
DATA_FOLDER = 'data_test'
FEATURES_FILE = 'features.pkl'
BENCHMARK_FOLDER = 'benchmark_results'
NUM_TEST_IMAGES = 10  # Number of images to use for testing

def load_or_build_database():
    """Load or build the features database"""
    if os.path.exists(FEATURES_FILE):
        print(f"Loading existing features database from {FEATURES_FILE}")
        with open(FEATURES_FILE, 'rb') as f:
            features_db = pickle.load(f)
        return features_db
    else:
        print("No features database found. Run app.py first to build the database.")
        return None

def extract_age_from_filename(filename):
    """Extract age from filename format like '35_1_0_20170109205304456.jpg' where 35 is the age"""
    try:
        # Extract the first number which should be the age
        parts = filename.split('_')
        if len(parts) >= 4:  # Check if filename has expected format
            age = int(parts[0])
            return age
        return None
    except Exception:
        return None

def categorize_age(age):
    """Categorize age into age groups"""
    if age < 13:
        return "child"
    elif age < 20:
        return "teen"
    elif age < 60:
        return "adult"
    else:
        return "senior"

def adjust_age_estimation(age, face_image):
    """Improved age estimation using multiple facial features and advanced image analysis"""
    # Convert age to float for precise calculations
    age = float(age)
    
    # Kiểm tra tuổi ban đầu từ DeepFace
    if age <= 0:
        age = 25  # Giá trị mặc định nếu DeepFace trả về tuổi không hợp lệ (tăng từ 15 lên 25)
    
    # In thông tin để debug
    print(f"Initial DeepFace age in benchmark: {age}")
    
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
    
    except Exception as e:
        print(f"Error in age estimation adjustments in benchmark: {e}")
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
    
    # Print debug info
    print(f"Final adjusted age in benchmark: {age}")
    
    return age

def extract_features(img_rgb):
    """Extract face features from an image, similar to app.py but standalone for benchmarking"""
    try:
        # Convert to array if it's not
        img_array = np.array(img_rgb)
        
        # Tiền xử lý ảnh để cải thiện chất lượng
        # 1. Chuyển sang grayscale để tăng độ tương phản
        gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # 2. Cải thiện độ tương phản
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray_image)
        
        # 3. Giảm nhiễu
        denoised = cv2.fastNlMeansDenoising(enhanced_gray, None, 10, 7, 21)
        
        # Tìm khuôn mặt bằng nhiều phương pháp
        # Thử với face_recognition trước
        face_locations = face_recognition.face_locations(img_array)
        
        # Nếu không tìm thấy khuôn mặt, thử với OpenCV cascade 
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
        
        # Nếu vẫn không tìm thấy mặt, thử phân tích cả ảnh
        if not face_locations:
            # Giả định một khuôn mặt ở chính giữa ảnh
            height, width = img_array.shape[:2]
            center_x, center_y = width // 2, height // 2
            face_size = min(width, height) // 2
            
            # Tạo một khuôn mặt ảo
            top = max(0, center_y - face_size)
            right = min(width, center_x + face_size)
            bottom = min(height, center_y + face_size)
            left = max(0, center_x - face_size)
            
            face_locations = [(top, right, bottom, left)]
            print(f"No face detected, using center of image")
        
        # Lấy encoding cho khuôn mặt
        face_encodings = face_recognition.face_encodings(img_array, face_locations)
        
        if not face_encodings and len(face_locations) > 0:
            # Mở rộng khu vực khuôn mặt nếu không lấy được encoding
            expanded_face_locations = []
            for (top, right, bottom, left) in face_locations:
                height, width = img_array.shape[:2]
                # Mở rộng 20% mỗi hướng
                expand_px = min(bottom - top, right - left) // 5
                new_top = max(0, top - expand_px)
                new_right = min(width, right + expand_px)
                new_bottom = min(height, bottom + expand_px)
                new_left = max(0, left - expand_px)
                expanded_face_locations.append((new_top, new_right, new_bottom, new_left))
                
            # Thử lấy encoding với khu vực mở rộng
            face_encodings = face_recognition.face_encodings(img_array, expanded_face_locations)
            if face_encodings:
                # Nếu thành công, cập nhật face_locations
                face_locations = expanded_face_locations
        
        # Nếu vẫn không lấy được encoding, sử dụng encoding giả
        if not face_encodings:
            dummy_encoding = np.zeros(128)
            random_factor = np.random.normal(0, 0.01, 128)
            dummy_encoding += random_factor
            dummy_encoding /= np.linalg.norm(dummy_encoding)  # Chuẩn hóa
            print("Could not extract encoding, using fallback")
            encoding = dummy_encoding
        else:
            encoding = face_encodings[0]
        
        # Trích xuất vùng khuôn mặt
        top, right, bottom, left = face_locations[0]
        face_image = img_array[top:bottom, left:right]
        
        try:
            # Phân tích với DeepFace
            try:
                analysis = DeepFace.analyze(face_image, actions=['age', 'emotion'], enforce_detection=False)
            except:
                # Nếu không thành công với face_image, thử với toàn bộ ảnh
                analysis = DeepFace.analyze(img_rgb, actions=['age', 'emotion'], enforce_detection=False)
            
            # Đảm bảo định dạng nhất quán
            if isinstance(analysis, list):
                analysis = analysis[0]
            
            # Trích xuất dữ liệu
            emotion = analysis["dominant_emotion"]
            raw_age = analysis["age"]
            
            # Cải thiện ước tính tuổi
            adjusted_age = adjust_age_estimation(raw_age, face_image)
            
            # Phân tích màu da
            skin_color = classify_skin_color(face_image)
            
            # Trả về kết quả
            features = {
                "encoding": encoding,
                "emotion": emotion,
                "age": adjusted_age,
                "age_group": categorize_age(adjusted_age),
                "skin_color": skin_color
            }
            
            return features
            
        except Exception as e:
            print(f"Error in DeepFace analysis: {e}")
            # Trả về kết quả mặc định nếu phân tích thất bại
            features = {
                "encoding": encoding,
                "emotion": "neutral",
                "age": 30,
                "age_group": "adult",
                "skin_color": "White"
            }
            return features
            
    except Exception as e:
        print(f"Critical error in feature extraction: {e}")
        return None

def classify_skin_color(face_image):
    """Classify skin color into White, Yellow, or Black based on HSV values"""
    try:
        # Chuyển sang HSV
        hsv_img = cv2.cvtColor(face_image, cv2.COLOR_RGB2HSV)
        
        # Kiểm tra hình ảnh đầu vào
        if hsv_img is None or hsv_img.size == 0:
            return "unknown"
        
        # Đảm bảo định dạng HSV
        if len(hsv_img.shape) < 3 or hsv_img.shape[2] != 3:
            return "unknown"
        
        # Áp dụng bộ lọc Gaussian để làm mịn trước khi phân tích
        smoothed_face = cv2.GaussianBlur(hsv_img, (5, 5), 0)
        
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
            center_region = hsv_img[hsv_img.shape[0]//4:3*hsv_img.shape[0]//4, 
                                  hsv_img.shape[1]//4:3*hsv_img.shape[1]//4]
            if center_region.size > 0:
                skin_pixels = center_region.reshape(-1, 3)
            else:
                skin_pixels = hsv_img.reshape(-1, 3)
        
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
        return None

def get_prefix_from_filename(filename):
    """Extract the prefix (e.g. 16_1_0_) from a filename"""
    # Assuming format like 16_1_0_20170109214419099.jpg
    parts = filename.split('_')
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}_{parts[2]}_"
    return ""

def benchmark_accuracy(file_dir, results_dir='benchmark_results'):
    # Create the results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create a text file to save results
    result_file_path = os.path.join(results_dir, 'accuracy_report.txt')
    result_file = open(result_file_path, 'w')
    
    # Create subdirectories for categorized results
    emotions_dir = os.path.join(results_dir, 'emotions')
    age_groups_dir = os.path.join(results_dir, 'age_groups')
    skin_colors_dir = os.path.join(results_dir, 'skin_colors')
    
    for subdir in [emotions_dir, age_groups_dir, skin_colors_dir]:
        if not os.path.exists(subdir):
            os.makedirs(subdir)
    
    # Initialize counters and lists for tracking accuracy
    emotion_actual = []
    emotion_predicted = []
    age_actual = []
    age_predicted = []
    age_group_actual = []
    age_group_predicted = []
    skin_color_actual = []
    skin_color_predicted = []
    total_faces = 0
    successful_faces = 0
    age_errors = []
    
    # Get all image files in the directory
    image_files = []
    for root, _, files in os.walk(file_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        result_file.write("No image files found for benchmarking.\n")
        result_file.close()
        return
    
    # Process each image file
    for img_path in image_files:
        filename = os.path.basename(img_path)
        
        # Extract actual emotion and age from filename if available
        try:
            actual_age = extract_age_from_filename(filename)
            if actual_age:
                actual_age = float(actual_age)
            else:
                continue  # Skip files without age information
                
            # Determine actual age group
            actual_age_group = categorize_age(actual_age)
            
            # Extract actual emotion (e.g., from folder name or filename)
            # This is a placeholder - implement based on your dataset structure
            folder_name = os.path.basename(os.path.dirname(img_path))
            actual_emotion = folder_name if folder_name in ['happy', 'sad', 'angry', 'fear', 'disgust', 'surprise', 'neutral'] else 'unknown'
            
            # Use the folder name or filename pattern to determine actual skin color
            # This is also a placeholder
            if 'white' in folder_name.lower():
                actual_skin_color = 'White'
            elif 'yellow' in folder_name.lower():
                actual_skin_color = 'Yellow'
            elif 'black' in folder_name.lower():
                actual_skin_color = 'Black'
            else:
                actual_skin_color = None  # Unknown or not specified
            
            # Run feature extraction on the image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            total_faces += 1
            try:
                features = extract_features(img_rgb)
                if features:
                    successful_faces += 1
                    predicted_emotion = features["emotion"]
                    predicted_age = features["age"]
                    predicted_age_group = categorize_age(predicted_age)
                    predicted_skin_color = features.get("skin_color", None)
                    
                    # Save for confusion matrix
                    if actual_emotion != 'unknown':
                        emotion_actual.append(actual_emotion)
                        emotion_predicted.append(predicted_emotion)
                    
                    age_actual.append(actual_age)
                    age_predicted.append(predicted_age)
                    
                    age_group_actual.append(actual_age_group)
                    age_group_predicted.append(predicted_age_group)
                    
                    if actual_skin_color and predicted_skin_color:
                        skin_color_actual.append(actual_skin_color)
                        skin_color_predicted.append(predicted_skin_color)
                    
                    # Calculate age error
                    age_error = abs(predicted_age - actual_age)
                    age_errors.append(age_error)
                    
                    # Save visualization
                    output_filename = f"{filename.split('.')[0]}_result.jpg"
                    result_img = visualize_benchmark_result(img_rgb, features, actual_age, actual_emotion, 
                                                          predicted_age, predicted_emotion, 
                                                          actual_skin_color, predicted_skin_color)
                    cv2.imwrite(os.path.join(results_dir, output_filename), 
                              cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
                    
                    # Save categorized results
                    output_emotion_dir = os.path.join(emotions_dir, predicted_emotion)
                    output_age_dir = os.path.join(age_groups_dir, predicted_age_group)
                    
                    for output_dir in [output_emotion_dir, output_age_dir]:
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                    
                    cv2.imwrite(os.path.join(output_emotion_dir, output_filename), 
                              cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(output_age_dir, output_filename), 
                              cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
                    
                    if predicted_skin_color:
                        output_skin_dir = os.path.join(skin_colors_dir, predicted_skin_color)
                        if not os.path.exists(output_skin_dir):
                            os.makedirs(output_skin_dir)
                        cv2.imwrite(os.path.join(output_skin_dir, output_filename), 
                                  cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
        except Exception as e:
            print(f"Error parsing file {img_path}: {e}")
    
    # Write summary to result file
    result_file.write(f"Total images processed: {total_faces}\n")
    result_file.write(f"Successful face detections: {successful_faces}\n")
    if total_faces > 0:
        result_file.write(f"Face detection rate: {successful_faces/total_faces*100:.2f}%\n")
    
    # Calculate and report age estimation accuracy
    if age_errors:
        mean_age_error = sum(age_errors) / len(age_errors)
        result_file.write(f"\nAge Estimation Accuracy:\n")
        result_file.write(f"Mean Absolute Error: {mean_age_error:.2f} years\n")
        
        # Group errors by age ranges
        age_ranges = [(0, 12, 'Children'), (13, 19, 'Teens'), (20, 39, 'Young Adults'), 
                     (40, 59, 'Middle-aged'), (60, 100, 'Seniors')]
        
        result_file.write("\nAge Estimation Accuracy by Age Group:\n")
        for age_min, age_max, group_name in age_ranges:
            group_errors = [error for error, actual in zip(age_errors, age_actual) 
                          if age_min <= actual <= age_max]
            if group_errors:
                group_mean_error = sum(group_errors) / len(group_errors)
                result_file.write(f"{group_name} ({age_min}-{age_max}): MAE = {group_mean_error:.2f} years, n = {len(group_errors)}\n")
    
    # Create confusion matrices for categorical predictions
    try:
        # 1. Emotion confusion matrix
        if len(set(emotion_actual)) > 1 and len(set(emotion_predicted)) > 1:
            emotion_cm = create_confusion_matrix(emotion_actual, emotion_predicted)
            result_file.write("\nEmotion Detection Accuracy:\n")
            result_file.write(f"Overall accuracy: {emotion_cm['accuracy']:.2f}%\n")
            for emotion in emotion_cm['per_class']:
                result_file.write(f"{emotion}: Precision = {emotion_cm['per_class'][emotion]['precision']:.2f}, " +
                               f"Recall = {emotion_cm['per_class'][emotion]['recall']:.2f}, " +
                               f"F1 = {emotion_cm['per_class'][emotion]['f1']:.2f}\n")
        else:
            result_file.write("\nEmotion Detection: Not enough distinct emotions for confusion matrix.\n")
        
        # 2. Age group confusion matrix
        if len(set(age_group_actual)) > 1 and len(set(age_group_predicted)) > 1:
            age_group_cm = create_confusion_matrix(age_group_actual, age_group_predicted)
            result_file.write("\nAge Group Classification Accuracy:\n")
            result_file.write(f"Overall accuracy: {age_group_cm['accuracy']:.2f}%\n")
            for age_group in age_group_cm['per_class']:
                result_file.write(f"{age_group}: Precision = {age_group_cm['per_class'][age_group]['precision']:.2f}, " +
                               f"Recall = {age_group_cm['per_class'][age_group]['recall']:.2f}, " +
                               f"F1 = {age_group_cm['per_class'][age_group]['f1']:.2f}\n")
        else:
            result_file.write("\nAge Group Classification: Not enough distinct age groups for confusion matrix.\n")
        
        # 3. Skin color confusion matrix
        if skin_color_actual and skin_color_predicted and len(set(skin_color_actual)) > 1 and len(set(skin_color_predicted)) > 1:
            skin_color_cm = create_confusion_matrix(skin_color_actual, skin_color_predicted)
            result_file.write("\nSkin Color Classification Accuracy:\n")
            result_file.write(f"Overall accuracy: {skin_color_cm['accuracy']:.2f}%\n")
            for skin_color in skin_color_cm['per_class']:
                result_file.write(f"{skin_color}: Precision = {skin_color_cm['per_class'][skin_color]['precision']:.2f}, " +
                               f"Recall = {skin_color_cm['per_class'][skin_color]['recall']:.2f}, " +
                               f"F1 = {skin_color_cm['per_class'][skin_color]['f1']:.2f}\n")
        else:
            result_file.write("\nSkin Color Classification: Not enough distinct skin colors for confusion matrix.\n")
            
    except Exception as e:
        result_file.write(f"\nError creating confusion matrices: {e}\n")
    
    result_file.close()
    print(f"Benchmark results saved to {results_dir}")

def create_confusion_matrix(actual, predicted):
    """Creates and returns a confusion matrix and classification metrics."""
    # Get unique classes
    classes = sorted(list(set(actual) | set(predicted)))
    
    # Check if there are valid classes
    if not classes:
        return {
            "accuracy": 0,
            "per_class": {},
            "confusion_matrix": []
        }
    
    # Initialize confusion matrix
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    
    # Fill confusion matrix
    for a, p in zip(actual, predicted):
        a_idx = classes.index(a)
        p_idx = classes.index(p)
        cm[a_idx, p_idx] += 1
    
    # Calculate metrics
    result = {
        "accuracy": 0,
        "per_class": {},
        "confusion_matrix": cm.tolist(),
        "classes": classes
    }
    
    # Overall accuracy
    total = np.sum(cm)
    correct = np.sum(np.diag(cm))
    if total > 0:
        result["accuracy"] = correct / total * 100
    
    # Per-class metrics
    for i, cls in enumerate(classes):
        true_pos = cm[i, i]
        false_pos = np.sum(cm[:, i]) - true_pos
        false_neg = np.sum(cm[i, :]) - true_pos
        
        # Calculate precision, recall, F1
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        result["per_class"][cls] = {
            "precision": precision * 100,
            "recall": recall * 100,
            "f1": f1 * 100
        }
    
    return result

def visualize_benchmark_result(img_rgb, features, actual_age, actual_emotion, 
                               predicted_age, predicted_emotion,
                               actual_skin_color=None, predicted_skin_color=None):
    """Visualize benchmark results for a single image"""
    h, w = img_rgb.shape[:2]
    
    # Create result visualization
    if h > 400 or w > 400:
        scale = min(400 / h, 400 / w)
        new_size = (int(w * scale), int(h * scale))
        img_rgb = cv2.resize(img_rgb, new_size)
    
    # Draw info on the image
    result_img = img_rgb.copy()
    
    # Calculate age difference
    age_diff = abs(predicted_age - actual_age) if actual_age else None
    
    # Create title with detection info
    result_texts = [
        f"Emotion: Actual: {actual_emotion}, Predicted: {predicted_emotion}",
        f"Age: Actual: {actual_age}, Predicted: {predicted_age:.1f}"
    ]
    
    if age_diff is not None:
        result_texts.append(f"Age Difference: {age_diff:.1f} years")
    
    if actual_skin_color and predicted_skin_color:
        result_texts.append(f"Skin Color: Actual: {actual_skin_color}, Predicted: {predicted_skin_color}")
    
    # Determine color for text based on accuracy
    emotion_color = (0, 255, 0) if actual_emotion == predicted_emotion else (0, 0, 255)
    age_color = (0, 255, 0) if age_diff and age_diff < 5 else (0, 0, 255)
    skin_color = (0, 255, 0) if actual_skin_color == predicted_skin_color else (0, 0, 255)
    
    # Create a white background for text
    text_bg = np.ones((100, result_img.shape[1], 3), dtype=np.uint8) * 255
    result_img = np.vstack([result_img, text_bg])
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_pos = result_img.shape[0] - 80
    
    # Write text lines
    cv2.putText(result_img, result_texts[0], (10, y_pos), font, 0.5, emotion_color, 1)
    cv2.putText(result_img, result_texts[1], (10, y_pos + 20), font, 0.5, age_color, 1)
    
    if age_diff is not None:
        cv2.putText(result_img, result_texts[2], (10, y_pos + 40), font, 0.5, age_color, 1)
    
    if actual_skin_color and predicted_skin_color:
        cv2.putText(result_img, result_texts[3], (10, y_pos + 60), font, 0.5, skin_color, 1)
    
    return result_img

def create_report(accuracy, emotion_acc, avg_age_diff, avg_processing_time, total_tests, skin_color_acc=None, age_accuracy_by_group=None):
    """Create a summary report of the benchmark"""
    skin_color_text = f"- Skin color detection accuracy: {skin_color_acc*100:.2f}%" if skin_color_acc is not None else ""
    
    # Add age accuracy by group to report if available
    age_group_text = ""
    if age_accuracy_by_group and age_accuracy_by_group:
        age_group_text = "\n\n## Age Estimation Accuracy by Group\n"
        for group, error in age_accuracy_by_group.items():
            age_group_text += f"- {group.capitalize()}: {error:.2f} years average error\n"
    
    report = f"""
    # Face Recognition System Benchmark Report

    ## Summary
    - Total test images: {total_tests}
    - Face recognition accuracy: {accuracy*100:.2f}%
    - Emotion detection accuracy: {emotion_acc*100:.2f}%
    {skin_color_text}
    - Average age difference: {avg_age_diff:.2f} years
    - Average processing time: {avg_processing_time:.3f} seconds
    {age_group_text}

    ## Details
    The benchmark tested the system's ability to correctly identify the same person
    in different images, as well as the accuracy of emotion detection and age estimation.
    
    ## Interpretation
    - **Face Recognition**: Measures how often the system correctly identified the same person
    - **Emotion Detection**: Measures if the emotion detected matches the emotion in the database
    - **Age Difference**: Average difference between estimated ages (lower is better)
    - **Processing Time**: Time taken to analyze each image and find matches
    """
    
    with open(os.path.join(BENCHMARK_FOLDER, 'benchmark_report.md'), 'w') as f:
        f.write(report)
    
    # Create a visualization of metrics
    plt.figure(figsize=(10, 6))
    metrics = ['Recognition Accuracy', 'Emotion Accuracy']
    values = [accuracy * 100, emotion_acc * 100]
    
    if skin_color_acc is not None:
        metrics.append('Skin Color Accuracy')
        values.append(skin_color_acc * 100)
    
    plt.bar(metrics, values, color=['#6e8efb', '#a777e3', '#ff9a9e'][:len(metrics)])
    plt.ylabel('Percentage (%)')
    plt.title('Face Recognition System Performance')
    plt.ylim(0, 100)
    
    for i, v in enumerate(values):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center')
    
    plt.savefig(os.path.join(BENCHMARK_FOLDER, 'performance_metrics.png'))
    plt.close()
    
    # Create age accuracy visualization if available
    if age_accuracy_by_group and len(age_accuracy_by_group) > 0:
        plt.figure(figsize=(10, 6))
        groups = list(age_accuracy_by_group.keys())
        errors = [age_accuracy_by_group[g] for g in groups]
        
        plt.bar(groups, errors, color=['#6e8efb', '#a777e3', '#ff9a9e', '#64c8ff'])
        plt.ylabel('Average Age Error (years)')
        plt.title('Age Estimation Error by Age Group')
        
        for i, v in enumerate(errors):
            plt.text(i, v + 0.5, f"{v:.1f}", ha='center')
        
        plt.savefig(os.path.join(BENCHMARK_FOLDER, 'age_accuracy_by_group.png'))
        plt.close()

def main():
    """Main function to run the benchmark"""
    parser = argparse.ArgumentParser(description='Run facial recognition benchmarks')
    parser.add_argument('--data_folder', type=str, default='data_test', 
                      help='Folder containing test images')
    parser.add_argument('--results_folder', type=str, default='benchmark_results',
                      help='Folder to store benchmark results')
    
    args = parser.parse_args()
    
    # Create results folder if it doesn't exist
    results_dir = args.results_folder
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    print("=" * 50)
    print("Starting benchmark process...")
    print(f"Data folder: {args.data_folder}")
    print(f"Results will be saved to: {results_dir}")
    print("=" * 50)
    
    # Run benchmark
    benchmark_accuracy(args.data_folder, results_dir)
    
    # Print results
    print(f"Benchmark completed!")
    print(f"Results saved to {results_dir}/ directory")
    print("=" * 50)

if __name__ == "__main__":
    main() 