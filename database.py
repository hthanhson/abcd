import os
import json
import numpy as np
import cv2
from feature_extraction import extract_features
import mysql.connector
from mysql.connector import Error
import pickle
from datetime import datetime
import scipy.spatial.distance as spatial
import db_config
import time

# Kết nối với database
def connect_to_database():
    """
    Connect to MySQL database with retry logic
    
    Returns:
        mysql.connector.connection: Database connection object or None if failed
    """
    retries = 0
    last_error = None
    
    while retries < db_config.DB_MAX_RETRIES:
        try:
            print(f"Connecting to database (attempt {retries + 1}/{db_config.DB_MAX_RETRIES})...")
            conn = mysql.connector.connect(
                host=db_config.DB_HOST,
                user=db_config.DB_USER,
                password=db_config.DB_PASSWORD,
                database=db_config.DB_NAME,
                port=db_config.DB_PORT,
                connection_timeout=db_config.DB_TIMEOUT
            )
            
            # Test the connection
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            
            print(f"Successfully connected to database {db_config.DB_NAME} on {db_config.DB_HOST}")
            return conn
            
        except Error as e:
            last_error = e
            print(f"Database connection attempt {retries + 1} failed: {e}")
            retries += 1
            
            if retries < db_config.DB_MAX_RETRIES:
                print(f"Retrying in {db_config.DB_RETRY_DELAY} seconds...")
                time.sleep(db_config.DB_RETRY_DELAY)
    
    print(f"All {db_config.DB_MAX_RETRIES} connection attempts failed. Last error: {last_error}")
    return None

def add_image_to_database(image_path, features):
    """
    Add image and its features to database
    
    Args:
        image_path: Path to the image file
        features: Dictionary containing extracted features
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Connect to database
        conn = connect_to_database()
        if conn is None:
            return False
        
        cursor = conn.cursor()
        
        # Insert into images table
        query = """
        INSERT INTO images (image_path, added_date)
        VALUES (%s, %s)
        """
        added_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(query, (image_path, added_date))
        
        # Get the image_id
        image_id = cursor.lastrowid
        
        # Insert into face_encodings table
        if features['face_encoding'] is not None:
            query = """
            INSERT INTO face_encodings (image_id, face_encoding)
            VALUES (%s, %s)
            """
            face_encoding_json = json.dumps(features['face_encoding'].tolist())
            cursor.execute(query, (image_id, face_encoding_json))
        
        # Insert into genders table
        if features['gender'] is not None:
            query = """
            INSERT INTO genders (image_id, gender_type, confidence_value)
            VALUES (%s, %s, %s)
            """
            cursor.execute(query, (image_id, features['gender'], features['gender_confidence']))
        
        # Insert into skin_colors table
        if features['skin_color'] is not None:
            query = """
            INSERT INTO skin_colors (image_id, skin_color_type)
            VALUES (%s, %s)
            """
            cursor.execute(query, (image_id, features['skin_color']))
        
        # Insert into emotions table
        if features['emotion'] is not None:
            query = """
            INSERT INTO emotions (image_id, emotion_type)
            VALUES (%s, %s)
            """
            cursor.execute(query, (image_id, features['emotion']))
        
        # Insert into feature_vectors table
        if features['gender_vector'] is not None and features['skin_vector'] is not None and \
           features['emotion_vector'] is not None and features['combined_vector'] is not None:
            query = """
            INSERT INTO feature_vectors (
                image_id, gender_vector, skin_vector, 
                emotion_vector, combined_vector
            )
            VALUES (%s, %s, %s, %s, %s)
            """
            
            gender_vector_json = json.dumps(features['gender_vector'].tolist())
            skin_vector_json = json.dumps(features['skin_vector'].tolist())
            emotion_vector_json = json.dumps(features['emotion_vector'].tolist())
            combined_vector_json = json.dumps(features['combined_vector'].tolist())
            
            cursor.execute(query, (
                image_id, 
                gender_vector_json, 
                skin_vector_json, 
                emotion_vector_json, 
                combined_vector_json
            ))
        
        # Commit changes
        conn.commit()
        
        # Close connection
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"Error adding image to database: {e}")
        return False

def delete_image_from_database(image_path):
    """
    Xóa ảnh và các đặc trưng tương ứng khỏi database
    
    Args:
        image_path: Đường dẫn đến ảnh cần xóa
        
    Returns:
        bool: True nếu thành công, False nếu thất bại
    """
    connection = connect_to_database()
    if connection is None:
        return False
    
    try:
        cursor = connection.cursor()
        
        # Tìm image_id dựa trên đường dẫn
        find_image_query = "SELECT id FROM images WHERE image_path = %s"
        cursor.execute(find_image_query, (image_path,))
        result = cursor.fetchone()
        
        if result is None:
            print(f"Image {image_path} not found in database")
            return False
        
        image_id = result[0]
        
        # Xóa các record liên quan từ các bảng con
        tables = ["feature_vectors", "face_encodings", "genders", "skin_colors", "emotions"]
        for table in tables:
            delete_query = f"DELETE FROM {table} WHERE image_id = %s"
            cursor.execute(delete_query, (image_id,))
        
        # Xóa record từ bảng images
        delete_image_query = "DELETE FROM images WHERE id = %s"
        cursor.execute(delete_image_query, (image_id,))
        
        connection.commit()
        return True
        
    except Error as e:
        print(f"Error deleting image from database: {e}")
        connection.rollback()
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def clear_database():
    """
    Xóa tất cả dữ liệu từ các bảng trong database
    
    Returns:
        bool: True nếu thành công, False nếu thất bại
    """
    try:
        # Connect to database
        conn = connect_to_database()
        if conn is None:
            return False
        
        cursor = conn.cursor()
        
        # Delete all data from tables
        tables = [
            'feature_vectors',
            'emotions',
            'skin_colors',
            'genders',
            'face_encodings',
            'images'
        ]
        
        for table in tables:
            query = f"DELETE FROM {table}"
            cursor.execute(query)
        
        # Reset auto-increment values
        for table in tables:
            query = f"ALTER TABLE {table} AUTO_INCREMENT = 1"
            cursor.execute(query)
        
        # Commit changes
        conn.commit()
        
        # Close connection
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"Error clearing database: {e}")
        return False

def build_database(folder_path):
    """
    Xây dựng database từ thư mục chứa ảnh
    
    Args:
        folder_path: Đường dẫn đến thư mục chứa ảnh
        
    Returns:
        tuple: (success, count) - Whether operation was successful and number of images processed
    """
    try:
        # Validate folder path
        if not os.path.isdir(folder_path):
            print(f"Invalid folder path: {folder_path}")
            return False, 0
        
        # Connect to database
        conn = connect_to_database()
        if conn is None:
            return False, 0
        
        # Get all image files in folder
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for root, _, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        
        print(f"Tìm thấy {len(image_files)} tệp ảnh")
        
        if len(image_files) == 0:
            print("Không tìm thấy tệp ảnh trong thư mục")
            return True, 0
        
        # Process each image
        count = 0
        for image_path in image_files:
            print(f"Đang xử lý {image_path}...")
            
            # Extract features
            features = extract_features(image_path=image_path)
            
            # Skip if no face found
            if not features['face_found']:
                print(f"Không tìm thấy khuôn mặt trong {image_path}")
                continue
            
            # Add to database
            success = add_image_to_database(image_path, features)
            
            if success:
                count += 1
                print(f"Đã thêm {image_path} vào cơ sở dữ liệu")
            else:
                print(f"Không thể thêm {image_path} vào cơ sở dữ liệu")
        
        # Close database connection
        conn.close()
        
        print(f"Đã xử lý thành công {count}/{len(image_files)} ảnh")
        return True, count
        
    except Exception as e:
        print(f"Lỗi khi xây dựng cơ sở dữ liệu: {e}")
        return False, 0

def compute_vector_distance(query_vector, db_vector, distance_type='euclidean'):
    """
    Compute distance between two vectors
    
    Args:
        query_vector: Query vector
        db_vector: Database vector
        distance_type: Type of distance to compute ('euclidean' or 'cosine')
        
    Returns:
        float: Distance between vectors
    """
    try:
        # Convert to numpy arrays if needed
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector)
        if not isinstance(db_vector, np.ndarray):
            db_vector = np.array(db_vector)
        
        # Make sure vectors have the same shape
        if query_vector.shape != db_vector.shape:
            print(f"Vector shapes do not match: {query_vector.shape} vs {db_vector.shape}")
            
            # Try to make them match
            min_length = min(query_vector.shape[0], db_vector.shape[0])
            query_vector = query_vector[:min_length]
            db_vector = db_vector[:min_length]
        
        # Check for NaN values
        if np.isnan(query_vector).any() or np.isnan(db_vector).any():
            print("Warning: NaN values detected in vectors, replacing with zeros")
            query_vector = np.nan_to_num(query_vector)
            db_vector = np.nan_to_num(db_vector)
        
        if distance_type == 'cosine':
            # Cosine similarity
            dot = np.dot(query_vector, db_vector)
            norm_query = np.linalg.norm(query_vector)
            norm_db = np.linalg.norm(db_vector)
            
            # Avoid division by zero
            if norm_query == 0 or norm_db == 0:
                return 1.0
            
            similarity = dot / (norm_query * norm_db)
            # Convert from similarity to distance (1 - similarity)
            return 1.0 - similarity
        else:
            # Euclidean distance
            return np.linalg.norm(query_vector - db_vector)
            
    except Exception as e:
        print(f"Error computing vector distance: {e}")
        return float('inf')  # Return infinite distance on error

def compute_vector_similarity(distance, max_distance=2.0, vector_type=None):
    """
    Convert distance to similarity score (0-1)
    
    Args:
        distance: Distance value
        max_distance: Maximum distance value for normalization
        vector_type: Type of vector ('combined', 'gender', 'skin', 'emotion')
        
    Returns:
        float: Similarity score (0-1)
    """
    try:
        # Check if distance is valid
        if distance != distance or distance > 1e9:  # Check for NaN or very large values
            print(f"WARNING: Invalid distance detected: {distance}")
            # Set a default value if distance is invalid
            distance = max_distance
        
        # Adjust max_distance based on vector type
        if vector_type == 'combined':
            # 176-dimensional vector can have a much larger distance
            adjusted_max_distance = 1000.0
        elif vector_type in ['gender', 'skin', 'emotion']:
            # 16-dimensional vectors
            adjusted_max_distance = 100.0
        else:
            # If not specified, use a middle ground
            adjusted_max_distance = 500.0
        
        # Compute similarity
        similarity = max(0, 1 - (distance / adjusted_max_distance))
        
        # Ensure similarity is not 0 even for large distances
        if similarity < 0.01:
            similarity = 0.01  # Set minimum to 1%
        
        return similarity
        
    except Exception as e:
        print(f"Error computing vector similarity: {e}")
        return 0.01  # Return minimum similarity on error

def compute_vector_distance_full(query_vector, db_vector, distance_type='euclidean'):
    """
    Compute distance between two 176-dimensional vectors
    
    Args:
        query_vector: Query vector (176-dimensional)
        db_vector: Database vector (176-dimensional)
        distance_type: Type of distance to compute ('euclidean' or 'cosine')
        
    Returns:
        float: Distance between vectors
    """
    # This function is the same as compute_vector_distance, but specifically for 176-dimensional vectors
    return compute_vector_distance(query_vector, db_vector, distance_type)

def find_similar_faces(query_features, top_n=3, filters=None):
    """
    Tìm kiếm khuôn mặt tương tự trong cơ sở dữ liệu chỉ dựa trên vector 176 chiều kết hợp
    
    Args:
        query_features: Đặc trưng của ảnh truy vấn
        top_n: Số lượng khuôn mặt tương tự cần trả về
        filters: Từ điển các bộ lọc để áp dụng (gender, skin_color, emotion)
        
    Returns:
        list: Danh sách các từ điển chứa thông tin về khuôn mặt tương tự
    """
    try:
        # Validate query features
        if query_features is None or 'combined_vector' not in query_features or query_features['combined_vector'] is None:
            print("Invalid query features")
            return []
        
        # Connect to database
        print("Connecting to database...")
        conn = connect_to_database()
        if conn is None:
            print("Failed to connect to database. Check db_config.py settings.")
            return []
        
        print("Database connection established")
        cursor = conn.cursor(dictionary=True)
        
        # Build the SQL query based on filters
        base_query = """
        SELECT i.id, i.image_path, g.gender_type, s.skin_color_type, e.emotion_type, 
               f.combined_vector
        FROM images i
        LEFT JOIN genders g ON i.id = g.image_id
        LEFT JOIN skin_colors s ON i.id = s.image_id
        LEFT JOIN emotions e ON i.id = e.image_id
        LEFT JOIN feature_vectors f ON i.id = f.image_id
        """
        
        where_clauses = []
        params = []
        
        # Add filter conditions if provided
        if filters:
            if 'gender' in filters and filters['gender']:
                where_clauses.append("g.gender_type = %s")
                params.append(filters['gender'])
            
            if 'skin_color' in filters and filters['skin_color']:
                where_clauses.append("s.skin_color_type = %s")
                params.append(filters['skin_color'])
            
            if 'emotion' in filters and filters['emotion']:
                where_clauses.append("e.emotion_type = %s")
                params.append(filters['emotion'])
        
        # Add WHERE clause if there are filters
        if where_clauses:
            base_query += " WHERE " + " AND ".join(where_clauses)
        
        # Execute query
        print(f"Executing query: {base_query} with params: {params}")
        try:
            cursor.execute(base_query, params)
            results = cursor.fetchall()
            print(f"Query executed. Found {len(results)} records.")
        except Exception as e:
            print(f"Database query error: {e}")
            conn.close()
            return []
        
        if len(results) == 0:
            print("No faces found in database.")
            conn.close()
            return []
        
        # Convert combined_vector from JSON to numpy array
        print("Processing database vectors...")
        valid_results = []
        for result in results:
            try:
                if result['combined_vector'] is not None:
                    result['combined_vector'] = np.array(json.loads(result['combined_vector']))
                    valid_results.append(result)
                else:
                    print(f"Warning: Null combined_vector for image {result['image_path']}")
            except Exception as e:
                print(f"Error processing vector for image {result['image_path']}: {e}")
        
        if len(valid_results) == 0:
            print("No valid vectors found in database results.")
            conn.close()
            return []
            
        print(f"Found {len(valid_results)} valid vectors.")
        
        # Compute distance between query vector and all database vectors
        # Chỉ sử dụng vector 176 chiều kết hợp (combined_vector)
        distances = []
        query_vector = query_features['combined_vector']
        
        print("Computing distances using 176-dimensional combined vector...")
        for i, result in enumerate(valid_results):
            # Skip comparing with same image if it exists in database
            if 'image_path' in query_features and query_features['image_path'] == result['image_path']:
                print(f"Skipping self-comparison with {result['image_path']}")
                continue
            
            try:
                # Sử dụng trực tiếp khoảng cách Euclidean giữa hai vector 176 chiều
                distance = np.linalg.norm(query_vector - result['combined_vector'])
                
                # Tính similarity dựa trên khoảng cách Euclidean
                # Càng gần thì similarity càng cao (1 = hoàn toàn giống nhau)
                max_distance = 100.0  # Giá trị chuẩn hóa
                similarity = max(0.0, 1.0 - (distance / max_distance))
                similarity = min(similarity, 1.0)  # Giới hạn trong khoảng [0, 1]
                
                # Add to distances list
                distances.append({
                    'index': i,
                    'distance': distance,
                    'similarity': similarity,
                    'rank': i + 1
                })
                
                print(f"Distance to {result['image_path']}: {distance}, similarity: {similarity}")
            except Exception as e:
                print(f"Error computing distance for {result['image_path']}: {e}")
        
        if len(distances) == 0:
            print("No distances computed. Cannot find similar faces.")
            conn.close()
            return []
        
        # Sort by distance (ascending order)
        distances.sort(key=lambda x: x['distance'])
        print(f"Distances sorted. Top distance: {distances[0]['distance']}")
        
        # Get top N results
        top_distances = distances[:top_n]
        top_indices = [d['index'] for d in top_distances]
        similar_faces = [valid_results[i] for i in top_indices]
        
        # Add distance information
        for i, face in enumerate(similar_faces):
            face['distance'] = top_distances[i]['distance']
            face['similarity'] = top_distances[i]['similarity'] 
            face['rank'] = i + 1  # Lưu thứ hạng để hiển thị (nếu cần)
            print(f"Top {i+1}: {face['image_path']} (distance: {face['distance']})")
        
        # Close database connection
        cursor.close()
        conn.close()
        
        return similar_faces
        
    except Exception as e:
        print(f"Error finding similar faces: {e}")
        import traceback
        traceback.print_exc()
        return []

def get_all_features():
    """
    Get all unique features from database for filtering
    
    Returns:
        dict: Dictionary containing lists of unique genders, skin colors, and emotions
    """
    try:
        # Connect to database
        conn = connect_to_database()
        if conn is None:
            return {
                'genders': [],
                'skin_colors': [],
                'emotions': []
            }
        
        cursor = conn.cursor()
        
        # Get unique genders
        cursor.execute("SELECT DISTINCT gender_type FROM genders")
        genders = [row[0] for row in cursor.fetchall()]
        
        # Get unique skin colors
        cursor.execute("SELECT DISTINCT skin_color_type FROM skin_colors")
        skin_colors = [row[0] for row in cursor.fetchall()]
        
        # Get unique emotions
        cursor.execute("SELECT DISTINCT emotion_type FROM emotions")
        emotions = [row[0] for row in cursor.fetchall()]
        
        # Close connection
        cursor.close()
        conn.close()
        
        return {
            'genders': genders,
            'skin_colors': skin_colors,
            'emotions': emotions
        }
        
    except Exception as e:
        print(f"Error getting all features: {e}")
        return {
            'genders': [],
            'skin_colors': [],
            'emotions': []
        }