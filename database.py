import os
import face_recognition
import numpy as np
import mysql.connector
import json
from mysql.connector import pooling, Error
from db_config import MYSQL_CONFIG
from feature_extraction import extract_features
from age_estimation import calibrate_age_estimation

class DatabaseManager:
    """
    Database manager class để quản lý kết nối
    """
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Singleton pattern để đảm bảo chỉ có một instance"""
        if cls._instance is None:
            cls._instance = DatabaseManager()
        return cls._instance
    
    def __init__(self):
        """Khởi tạo connection pool"""
        try:
            self.connection_pool = pooling.MySQLConnectionPool(
                pool_name="face_pool",
                pool_size=5,
                **MYSQL_CONFIG
            )
            print("MySQL connection pool created successfully")
        except Error as e:
            print(f"Error creating connection pool: {e}")
            self.connection_pool = None
    
    def execute_query(self, query, params=None, fetch_one=False, fetch_all=False, dictionary=True):
        """
        Thực thi truy vấn với xử lý kết nối tự động
        
        Args:
            query (str): Truy vấn SQL cần thực thi
            params (tuple/list/dict): Tham số cho truy vấn
            fetch_one (bool): Lấy một kết quả
            fetch_all (bool): Lấy tất cả kết quả
            dictionary (bool): Trả về kết quả dưới dạng dictionary
            
        Returns:
            Kết quả truy vấn tùy theo fetch_one, fetch_all, hoặc lastrowid cho thao tác INSERT
        """
        if self.connection_pool is None:
            print("Database connection pool not initialized")
            return None
            
        conn = None
        try:
            conn = self.connection_pool.get_connection()
            cursor = conn.cursor(dictionary=dictionary)
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            result = None
            
            if fetch_one:
                result = cursor.fetchone()
            elif fetch_all:
                result = cursor.fetchall()
            else:
                # For INSERT operations, return last inserted ID
                conn.commit()
                result = cursor.lastrowid
                
            cursor.close()
            conn.close()
            return result
            
        except Error as e:
            print(f"Database error: {e}")
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
                cursor.close()
                conn.close()
            return None
    
    def execute_transaction(self, queries_with_params):
        """
        Thực thi nhiều truy vấn trong một giao dịch
        
        Args:
            queries_with_params (list): Danh sách các tuple (query, params)
            
        Returns:
            bool: Thành công hay thất bại
        """
        if self.connection_pool is None:
            print("Database connection pool not initialized")
            return False
            
        conn = None
        try:
            conn = self.connection_pool.get_connection()
            cursor = conn.cursor()
            
            for query, params in queries_with_params:
                cursor.execute(query, params)
                
            conn.commit()
            cursor.close()
            conn.close()
            return True
            
        except Error as e:
            print(f"Transaction error: {e}")
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
                cursor.close()
                conn.close()
            return False
    
    def get_image_count(self):
        """Lấy số lượng ảnh trong database"""
        result = self.execute_query(
            "SELECT COUNT(*) as count FROM images",
            fetch_one=True
        )
        return result['count'] if result else 0

# Tạo instance toàn cục
db_manager = DatabaseManager.get_instance()

def initialize_database():
    """Kết nối đến cơ sở dữ liệu MySQL và kiểm tra kết nối"""
    if db_manager.connection_pool is None:
        print("Connection pool not initialized. Check your MySQL configuration.")
        return None
    
    try:
        image_count = db_manager.get_image_count()
        print(f"MySQL database connected successfully. {image_count} images in database.")
        
        # Để tương thích với code cũ
        return {}
    
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def add_image_to_database(image_path, encoding, emotion, age, age_group, skin_color, emotion_confidence=1.0):
    """Thêm ảnh và đặc trưng vào cơ sở dữ liệu MySQL"""
    if db_manager.connection_pool is None:
        print("Connection pool not initialized")
        return None
    
    try:
        # 1. Thêm ảnh vào bảng images
        image_name = os.path.basename(image_path)
        image_id = db_manager.execute_query(
            """
            INSERT INTO images (image_name, image_path) 
            VALUES (%s, %s) 
            ON DUPLICATE KEY UPDATE image_id=LAST_INSERT_ID(image_id)
            """,
            (image_name, image_path)
        )
        
        # Nếu không có image_id, lấy id của ảnh đã tồn tại
        if not image_id:
            result = db_manager.execute_query(
                "SELECT image_id FROM images WHERE image_path = %s",
                (image_path,),
                fetch_one=True
            )
            image_id = result['image_id'] if result else None
            
        if not image_id:
            print(f"Could not insert or retrieve image ID for {image_path}")
            return None
            
        # Chuẩn bị queries cho transaction
        encoding_json = json.dumps(encoding.tolist() if isinstance(encoding, np.ndarray) else encoding)
        
        queries = [
            # Xóa face encodings cũ
            ("DELETE FROM face_encodings WHERE image_id = %s", (image_id,)),
            # Thêm encoding mới
            ("INSERT INTO face_encodings (image_id, encoding_data) VALUES (%s, %s)", (image_id, encoding_json)),
            # Xóa cảm xúc cũ
            ("DELETE FROM emotions WHERE image_id = %s", (image_id,)),
            # Thêm cảm xúc mới
            ("INSERT INTO emotions (image_id, emotion_type, confidence_value) VALUES (%s, %s, %s)", (image_id, emotion, emotion_confidence)),
            # Xóa tuổi cũ
            ("DELETE FROM ages WHERE image_id = %s", (image_id,)),
            # Thêm tuổi mới
            ("INSERT INTO ages (image_id, estimated_age, confidence_value) VALUES (%s, %s, %s)", (image_id, float(age), 1.0)),
            # Xóa nhóm tuổi cũ
            ("DELETE FROM age_groups WHERE image_id = %s", (image_id,)),
            # Thêm nhóm tuổi mới
            ("INSERT INTO age_groups (image_id, age_group, confidence_value) VALUES (%s, %s, %s)", (image_id, age_group, 1.0)),
            # Xóa màu da cũ
            ("DELETE FROM skin_colors WHERE image_id = %s", (image_id,)),
            # Thêm màu da mới
            ("INSERT INTO skin_colors (image_id, skin_color, confidence_value) VALUES (%s, %s, %s)", (image_id, skin_color, 1.0))
        ]
        
        # Thực hiện transaction
        success = db_manager.execute_transaction(queries)
        
        if success:
            print(f"Image {image_name} added to database with ID {image_id}")
            return image_id
        else:
            print(f"Failed to add image {image_name} to database")
            return None
        
    except Exception as e:
        print(f"Error adding image to database: {e}")
        return None

def get_all_encodings():
    """Lấy tất cả encodings từ cơ sở dữ liệu"""
    if db_manager.connection_pool is None:
        return [], []
    
    try:
        # Lấy cả image_id và encoding_data từ bảng face_encodings
        rows = db_manager.execute_query(
            """
            SELECT fe.image_id, fe.encoding_data, i.image_path 
            FROM face_encodings fe
            JOIN images i ON fe.image_id = i.image_id
            """,
            fetch_all=True
        )
        
        if not rows:
            return [], []
            
        encodings = []
        image_paths = []
        
        for row in rows:
            # Chuyển chuỗi JSON thành mảng NumPy
            encoding_list = json.loads(row['encoding_data'])
            encoding = np.array(encoding_list)
            
            encodings.append(encoding)
            image_paths.append(row['image_path'])
        
        return np.array(encodings), image_paths
    
    except Exception as e:
        print(f"Error getting encodings: {e}")
        return [], []

def find_similar_faces(query_encoding, top_n=3, query_emotion=None, query_age=None, query_age_group=None, query_skin_color=None):
    """Tìm top N khuôn mặt tương tự dựa trên encoding và các đặc trưng khác"""
    if db_manager.connection_pool is None:
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
        
        # Danh sách ứng viên với thông tin chi tiết
        candidates = []
        
        for i in candidate_indices:
            image_path = image_paths[i]
            face_similarity = 1 - face_distances[i]  # Chuyển khoảng cách thành độ tương tự
            
            # Lấy thông tin chi tiết của ảnh
            face_info = db_manager.execute_query(
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
                (image_path,),
                fetch_one=True
            )
            
            if face_info:
                # Tính toán các điểm tương đồng cho từng đặc trưng
                
                # 1. Điểm cơ bản từ face embedding (0-1)
                # Điểm tương đồng thô (raw) có thể > 0.5 nên cần chuẩn hóa
                # Đảm bảo base_score không vượt quá 0.5 (50%)
                base_score = min(face_similarity, 1.0) * 0.5
                
                # 2. Điểm cảm xúc - nếu cùng cảm xúc thì có 0.25 điểm
                emotion_score = 0.0
                if query_emotion and face_info['emotion'] == query_emotion:
                    emotion_score = 0.25
                
                # 3. Điểm độ tuổi - dựa trên khoảng cách tuổi tương đối
                age_score = 0.0
                if query_age is not None and face_info['age'] is not None:
                    age_diff = abs(float(face_info['age']) - float(query_age))
                    max_age_diff = 50.0  # Giả sử chênh lệch tuổi tối đa
                    age_score = max(0, 0.15 * (1 - age_diff / max_age_diff))
                
                # 4. Điểm màu da - nếu cùng màu da thì có 0.1 điểm
                skin_score = 0.0
                if query_skin_color and face_info['skin_color'] == query_skin_color:
                    skin_score = 0.1
                
                # Tính tổng điểm - phân bổ trọng số:
                # - Face embedding: 50%
                # - Cảm xúc: 25%
                # - Độ tuổi: 15%
                # - Màu da: 10%
                total_score = base_score + emotion_score + age_score + skin_score
                
                # Đảm bảo tổng điểm không vượt quá 1.0 (100%)
                total_score = min(total_score, 1.0)
                
                # Lưu lại thông tin về ảnh và các thành phần điểm
                candidates.append({
                    'image_path': face_info['image_path'],
                    'total_score': float(total_score),         # Điểm tổng hợp tất cả đặc trưng
                    'emotion': face_info['emotion'],
                    'age': face_info['age'],
                    'age_group': face_info['age_group'],
                    'skin_color': face_info['skin_color'],
                    # Lưu thêm các thành phần điểm để in ra terminal
                    'base_score': float(base_score),
                    'emotion_score': float(emotion_score),
                    'age_score': float(age_score),
                    'skin_score': float(skin_score)
                })
        
        # Sắp xếp lại theo điểm tổng hợp và chọn top_n kết quả
        candidates.sort(key=lambda x: x['total_score'], reverse=True)
        results = candidates[:top_n]
        
        # In thông tin chi tiết về các thành phần điểm của top N kết quả
        print("\n=== DETAILED SCORING BREAKDOWN FOR TOP RESULTS ===")
        for i, result in enumerate(results):
            print(f"\n--- RESULT #{i+1}: {os.path.basename(result['image_path'])} ---")
            print(f"base_score (50%): {result.get('base_score', 'N/A')}")
            print(f"emotion_score (25%): {result.get('emotion_score', 'N/A')}")
            print(f"age_score (15%): {result.get('age_score', 'N/A')}")
            print(f"skin_score (10%): {result.get('skin_score', 'N/A')}")
            print(f"TOTAL SCORE: {result.get('total_score', 'N/A')}")
        print("===================================================\n")
        
        # Giữ nguyên các trường để frontend có thể sử dụng
        for result in results:
            # Không xóa raw_similarity và total_score để frontend có thể hiển thị
            pass
        
        return results
    
    except Exception as e:
        print(f"Error finding similar faces: {e}")
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
    if db_manager.connection_pool is None:
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
        # Tạm thời tắt foreign key constraints để xóa dữ liệu
        db_manager.execute_query("SET FOREIGN_KEY_CHECKS = 0")
        
        # Xóa dữ liệu từ tất cả các bảng
        for table in tables:
            db_manager.execute_query(f"TRUNCATE TABLE {table}")
            print(f"Cleared data from table: {table}")
        
        # Bật lại foreign key constraints
        db_manager.execute_query("SET FOREIGN_KEY_CHECKS = 1")
        
        print("All data has been cleared from the database")
        return True
        
    except Exception as e:
        # Đảm bảo bật lại foreign key constraints nếu có lỗi
        db_manager.execute_query("SET FOREIGN_KEY_CHECKS = 1")
        print(f"Error clearing database: {e}")
        return False 