import os
import face_recognition
import numpy as np
import mysql.connector
import json
from mysql.connector import pooling, Error
from db_config import MYSQL_CONFIG
from feature_extraction import extract_features
from age_estimation import calibrate_age_estimation

# Tạo connection pool
try:
    connection_pool = pooling.MySQLConnectionPool(
        pool_name="face_pool",
        pool_size=5,
        **MYSQL_CONFIG
    )
    print("MySQL connection pool created successfully")
except Error as e:
    print(f"Error creating connection pool: {e}")
    connection_pool = None

def initialize_database():
    """Kết nối đến cơ sở dữ liệu MySQL và kiểm tra kết nối"""
    if connection_pool is None:
        print("Connection pool not initialized. Check your MySQL configuration.")
        return None
    
    try:
        conn = connection_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Kiểm tra kết nối
        cursor.execute("SELECT COUNT(*) as count FROM images")
        result = cursor.fetchone()
        image_count = result['count'] if result else 0
        
        cursor.close()
        conn.close()
        
        print(f"MySQL database connected successfully. {image_count} images in database.")
        
        # Để tương thích với code cũ
        return {}
    
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def add_image_to_database(image_path, encoding, emotion, age, age_group, skin_color, emotion_confidence=1.0):
    """Thêm ảnh và đặc trưng vào cơ sở dữ liệu MySQL"""
    if connection_pool is None:
        print("Connection pool not initialized")
        return None
    
    conn = None
    try:
        conn = connection_pool.get_connection()
        cursor = conn.cursor()
        
        # 1. Thêm ảnh vào bảng images
        image_name = os.path.basename(image_path)
        cursor.execute(
            """
            INSERT INTO images (image_name, image_path) 
            VALUES (%s, %s) 
            ON DUPLICATE KEY UPDATE image_id=LAST_INSERT_ID(image_id)
            """,
            (image_name, image_path)
        )
        
        # Lấy ID của ảnh vừa thêm
        image_id = cursor.lastrowid
        if not image_id:
            cursor.execute("SELECT image_id FROM images WHERE image_path = %s", (image_path,))
            image_id = cursor.fetchone()[0]
            
        # 2. Thêm face encoding
        # Xóa encodings cũ nếu có
        cursor.execute("DELETE FROM face_encodings WHERE image_id = %s", (image_id,))
        
        # Chuyển encoding thành chuỗi JSON
        encoding_json = json.dumps(encoding.tolist() if isinstance(encoding, np.ndarray) else encoding)
        
        # Thêm encoding mới dưới dạng chuỗi JSON
        cursor.execute(
            "INSERT INTO face_encodings (image_id, encoding_data) VALUES (%s, %s)",
            (image_id, encoding_json)
        )
        
        # 3. Thêm cảm xúc - xóa cũ nếu có
        cursor.execute("DELETE FROM emotions WHERE image_id = %s", (image_id,))
        cursor.execute(
            "INSERT INTO emotions (image_id, emotion_type, confidence_value) VALUES (%s, %s, %s)",
            (image_id, emotion, emotion_confidence)
        )
        
        # 4. Thêm tuổi - xóa cũ nếu có
        cursor.execute("DELETE FROM ages WHERE image_id = %s", (image_id,))
        cursor.execute(
            "INSERT INTO ages (image_id, estimated_age, confidence_value) VALUES (%s, %s, %s)",
            (image_id, float(age), 1.0)
        )
        
        # 5. Thêm nhóm tuổi - xóa cũ nếu có
        cursor.execute("DELETE FROM age_groups WHERE image_id = %s", (image_id,))
        cursor.execute(
            "INSERT INTO age_groups (image_id, age_group, confidence_value) VALUES (%s, %s, %s)",
            (image_id, age_group, 1.0)
        )
        
        # 6. Thêm màu da - xóa cũ nếu có
        cursor.execute("DELETE FROM skin_colors WHERE image_id = %s", (image_id,))
        cursor.execute(
            "INSERT INTO skin_colors (image_id, skin_color, confidence_value) VALUES (%s, %s, %s)",
            (image_id, skin_color, 1.0)
        )
        
        conn.commit()
        print(f"Image {image_name} added to database with ID {image_id}")
        
        cursor.close()
        conn.close()
        return image_id
        
    except Error as e:
        print(f"Error adding image to database: {e}")
        if conn:
            try:
                conn.rollback()
            except:
                pass
            cursor.close()
            conn.close()
        return None

def get_all_encodings():
    """Lấy tất cả encodings từ cơ sở dữ liệu"""
    if connection_pool is None:
        return [], []
    
    try:
        conn = connection_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Lấy cả image_id và encoding_data từ bảng face_encodings
        cursor.execute("""
            SELECT fe.image_id, fe.encoding_data, i.image_path 
            FROM face_encodings fe
            JOIN images i ON fe.image_id = i.image_id
        """)
        
        rows = cursor.fetchall()
        encodings = []
        image_paths = []
        
        for row in rows:
            # Chuyển chuỗi JSON thành mảng NumPy
            encoding_list = json.loads(row['encoding_data'])
            encoding = np.array(encoding_list)
            
            encodings.append(encoding)
            image_paths.append(row['image_path'])
        
        cursor.close()
        conn.close()
        
        return np.array(encodings), image_paths
    
    except Error as e:
        print(f"Error getting encodings: {e}")
        return [], []

def find_similar_faces(query_encoding, top_n=3, query_emotion=None, query_age=None, query_age_group=None, query_skin_color=None):
    """Tìm top N khuôn mặt tương tự dựa trên encoding và các đặc trưng khác"""
    if connection_pool is None:
        return []
    
    try:
        # Lấy tất cả encodings và image_paths từ database
        encodings, image_paths = get_all_encodings()
        
        # Kiểm tra xem encodings có phải là mảng rỗng không
        if isinstance(encodings, list) or (isinstance(encodings, np.ndarray) and encodings.size == 0):
            return []
        
        # Kiểm tra xem query_encoding có hợp lệ không
        if query_encoding is None or not isinstance(query_encoding, np.ndarray):
            print("Invalid query encoding")
            return []
            
        # Đảm bảo các encoding có cùng kích thước
        if len(encodings.shape) != 2 or encodings.shape[1] != 128 or query_encoding.shape[0] != 128:
            print(f"Incompatible encoding dimensions: encodings shape {encodings.shape}, query shape {query_encoding.shape}")
            return []
        
        # Tính khoảng cách giữa query_encoding và tất cả encodings
        try:
            face_distances = face_recognition.face_distance(encodings, query_encoding)
        except Exception as e:
            print(f"Error calculating face distances: {e}")
            return []
        
        # Lấy nhiều ảnh hơn số lượng cần trả về cuối cùng để có thể lọc tiếp
        candidates_count = min(len(encodings), top_n * 3)  # Lấy top_n*3 ứng viên hoặc tất cả nếu ít hơn
        
        # Sắp xếp và lấy các ứng viên tiềm năng
        candidate_indices = np.argsort(face_distances)[:candidates_count]
        
        conn = connection_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Danh sách ứng viên với thông tin chi tiết
        candidates = []
        
        for i in candidate_indices:
            image_path = image_paths[i]
            face_similarity = 1 - face_distances[i]  # Chuyển khoảng cách thành độ tương tự
            
            # Lấy thông tin chi tiết của ảnh
            cursor.execute(
                """
                SELECT 
                    i.image_path,
                    e.emotion_type AS emotion,
                    a.estimated_age AS age,
                    ag.age_group,
                    s.skin_color
                FROM images i
                LEFT JOIN emotions e ON i.image_id = e.image_id
                LEFT JOIN ages a ON i.image_id = a.image_id
                LEFT JOIN age_groups ag ON i.image_id = ag.image_id
                LEFT JOIN skin_colors s ON i.image_id = s.image_id
                WHERE i.image_path = %s
                """,
                (image_path,)
            )
            
            face_info = cursor.fetchone()
            if face_info:
                # Tính toán các điểm tương đồng cho từng đặc trưng
                
                # 1. Điểm cơ bản từ face embedding (0-1)
                base_score = face_similarity
                
                # 2. Điểm cảm xúc - nếu cùng cảm xúc thì có 0.2 điểm
                emotion_score = 0.0
                if query_emotion and face_info['emotion'] == query_emotion:
                    emotion_score = 0.2
                
                # 3. Điểm độ tuổi - dựa trên khoảng cách tuổi tương đối
                age_score = 0.0
                if query_age is not None and face_info['age'] is not None:
                    age_diff = abs(float(face_info['age']) - float(query_age))
                    max_age_diff = 50.0  # Giả sử chênh lệch tuổi tối đa
                    age_score = max(0, 0.15 * (1 - age_diff / max_age_diff))
                
                # 4. Điểm nhóm tuổi - nếu cùng nhóm tuổi thì có 0.1 điểm
                age_group_score = 0.0
                if query_age_group and face_info['age_group'] == query_age_group:
                    age_group_score = 0.1
                
                # 5. Điểm màu da - nếu cùng màu da thì có 0.05 điểm
                skin_score = 0.0
                if query_skin_color and face_info['skin_color'] == query_skin_color:
                    skin_score = 0.05
                
                # Tính tổng điểm - face embedding vẫn chiếm trọng số lớn nhất (50%)
                # Các đặc trưng khác chiếm 50% còn lại
                total_score = base_score * 0.5 + emotion_score + age_score + age_group_score + skin_score
                
                candidates.append({
                    'image_path': face_info['image_path'],
                    'similarity': float(face_similarity),  # Điểm tương đồng dựa chỉ trên khuôn mặt
                    'total_score': float(total_score),     # Điểm tổng hợp tất cả đặc trưng
                    'emotion': face_info['emotion'],
                    'age': face_info['age'],
                    'age_group': face_info['age_group'],
                    'skin_color': face_info['skin_color']
                })
        
        cursor.close()
        conn.close()
        
        # Sắp xếp lại theo điểm tổng hợp và chọn top_n kết quả
        candidates.sort(key=lambda x: x['total_score'], reverse=True)
        results = candidates[:top_n]
        
        # Cập nhật lại trường similarity để frontend hiểu được
        for result in results:
            result['similarity'] = result['total_score']
            del result['total_score']  # Xóa trường tạm thời
        
        return results
    
    except Error as e:
        print(f"Error finding similar faces: {e}")
        return []

def filter_database(emotion=None, min_age=0, max_age=100, age_group=None, skin_color=None):
    """Lọc cơ sở dữ liệu theo các tiêu chí"""
    if connection_pool is None:
        return []
    
    try:
        conn = connection_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        query = """
            SELECT 
                i.image_path,
                e.emotion_type AS emotion,
                a.estimated_age AS age,
                ag.age_group,
                s.skin_color
            FROM images i
            LEFT JOIN emotions e ON i.image_id = e.image_id
            LEFT JOIN ages a ON i.image_id = a.image_id
            LEFT JOIN age_groups ag ON i.image_id = ag.image_id
            LEFT JOIN skin_colors s ON i.image_id = s.image_id
            WHERE 1=1
        """
        
        params = []
        
        if emotion and emotion != '':
            query += " AND e.emotion_type = %s"
            params.append(emotion)
        
        if min_age is not None and max_age is not None:
            query += " AND a.estimated_age BETWEEN %s AND %s"
            params.append(min_age)
            params.append(max_age)
        
        if age_group and age_group != '':
            query += " AND ag.age_group = %s"
            params.append(age_group)
        
        if skin_color and skin_color != '':
            query += " AND s.skin_color = %s"
            params.append(skin_color)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        return results
    
    except Error as e:
        print(f"Error filtering database: {e}")
        return []

def build_database(data_folder):
    """Xây dựng cơ sở dữ liệu từ thư mục ảnh"""
    # Hiệu chỉnh ước tính tuổi nếu có thể
    print("Calibrating age estimation...")
    calibrate_age_estimation(data_folder)
    
    # Lấy danh sách file ảnh
    image_files = [f for f in os.listdir(data_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total = len(image_files)
    processed_count = 0
    
    print(f"Processing {total} images...")
    
    for i, img_file in enumerate(image_files):
        print(f"Processing image {i+1}/{total}: {img_file}")
        img_path = os.path.join(data_folder, img_file)
        
        # Trích xuất đặc trưng
        encoding, emotion, age, age_group, skin_color = extract_features(img_path)
        
        if encoding is not None:
            # Thêm vào cơ sở dữ liệu
            image_id = add_image_to_database(img_path, encoding, emotion, age, age_group, skin_color)
            
            if image_id:
                processed_count += 1
    
    print(f"Database built with {processed_count} faces")
    return processed_count

def clear_database():
    """Xóa toàn bộ dữ liệu trong tất cả các bảng"""
    if connection_pool is None:
        print("Connection pool not initialized")
        return False
    
    # Danh sách các bảng theo thứ tự xóa (từ bảng con đến bảng cha)
    tables = [
        'face_encodings',
        'emotions',
        'ages',
        'age_groups',
        'skin_colors',
        'images'
    ]
    
    try:
        conn = connection_pool.get_connection()
        cursor = conn.cursor()
        
        # Tạm thời tắt foreign key constraints để xóa dữ liệu
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
        
        # Xóa dữ liệu từ tất cả các bảng
        for table in tables:
            cursor.execute(f"TRUNCATE TABLE {table}")
            print(f"Cleared data from table: {table}")
        
        # Bật lại foreign key constraints
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
        
        cursor.close()
        conn.close()
        
        print("All data has been cleared from the database")
        return True
        
    except Error as e:
        print(f"Error clearing database: {e}")
        if conn:
            try:
                # Bật lại foreign key constraints nếu có lỗi
                cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
                conn.close()
            except:
                pass
        return False 