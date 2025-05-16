import os
import json
import numpy as np
import cv2
from feature_extraction import extract_features
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import scipy.spatial.distance as spatial
import db_config
import time

# Thêm hàm trợ giúp kiểm tra kết nối
def is_connection_valid(connection):
    """
    Kiểm tra xem kết nối MySQL có hợp lệ và kết nối hay không
    
    Args:
        connection: Đối tượng kết nối MySQL
        
    Returns:
        bool: True nếu kết nối hợp lệ và đang kết nối, False nếu không
    """
    if connection is None:
        return False
    
    try:
        return connection.is_connected()
    except:
        return False
        
# Điều chỉnh hàm connect_to_database để đảm bảo phần xử lý lỗi và đóng kết nối được xử lý đúng
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
            
            # Test the connection with direct query
            conn.cmd_query("SELECT 1")
            # Lấy kết quả (không nhất thiết phải lưu)
            conn.get_rows()
            
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

def execute_query(connection, query, params=None):
    """
   
    Args:
        connection: Kết nối MySQL
        query: Câu truy vấn SQL
        params: Tham số cho truy vấn (tuple)
        
    Returns:
        list: Kết quả cho truy vấn SELECT/SHOW
        bool: True cho các truy vấn INSERT/UPDATE/DELETE thành công
        None: Nếu có lỗi xảy ra
    """
    # Kiểm tra kết nối có hợp lệ không
    if not is_connection_valid(connection):
        print("Connection is invalid or not connected, cannot execute query")
        return None
    
    try:
        # Chuẩn bị câu truy vấn với tham số
        if params:
            prepared_query = query
            for param in params:
                # Xử lý các loại dữ liệu khác nhau
                if isinstance(param, str):
                    param_value = f"'{param}'"
                elif param is None:
                    param_value = 'NULL'
                else:
                    param_value = str(param)
                
                # Thay thế %s đầu tiên bằng giá trị tham số
                prepared_query = prepared_query.replace('%s', param_value, 1)
        else:
            prepared_query = query
        
        # Xác định loại truy vấn
        is_select = prepared_query.strip().upper().startswith('SELECT')
        is_show = prepared_query.strip().upper().startswith('SHOW')
        
        # Thực hiện truy vấn trực tiếp thông qua connection
        connection.cmd_query(prepared_query)
        
        if is_select or is_show:
            result = connection.get_rows()
            if result:
                rows = result[0]  # get_rows() trả về tuple (rows, eof)
                return rows
            return []
            
        # Đối với các truy vấn INSERT, UPDATE, DELETE, chỉ trả về True nếu thành công
        # (chúng ta đã đến đây mà không có lỗi nào, nên truy vấn đã thành công)
        return True
        
    except Error as e:
        print(f"Error executing query: {e}")
        print(f"Query: {query}")
        if params:
            print(f"Params: {params}")
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
        
        # Insert into images table
        query = f"""
        INSERT INTO images (image_path, added_date)
        VALUES ('{image_path}', '{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        """
        
        # Thực hiện truy vấn
        insert_result = execute_query(conn, query)
        if insert_result is None:  # Nếu insert thất bại
            print("Failed to insert image into database")
            conn.close()
            return False
        
        # Get the image_id - MySQL sẽ trả về ID cuối cùng được chèn vào
        query_id = "SELECT LAST_INSERT_ID()"
        result = execute_query(conn, query_id)
        if not result or len(result) == 0:
            print("Failed to get last insert ID")
            conn.close()
            return False
            
        image_id = result[0][0]
        
        # Insert into face_encodings table
        if features['face_encoding'] is not None:
            face_encoding_json = json.dumps(features['face_encoding'].tolist())
            query = f"""
            INSERT INTO face_encodings (image_id, face_encoding)
            VALUES ({image_id}, '{face_encoding_json}')
            """
            if execute_query(conn, query) is None:
                print("Failed to insert face encoding")
                conn.close()
                return False
        
        # Insert into genders table
        if features['gender'] is not None:
            query = f"""
            INSERT INTO genders (image_id, gender_type, confidence_value)
            VALUES ({image_id}, '{features['gender']}', {features['gender_confidence']})
            """
            if execute_query(conn, query) is None:
                print("Failed to insert gender")
                conn.close()
                return False
        
        # Insert into skin_colors table
        if features['skin_color'] is not None:
            query = f"""
            INSERT INTO skin_colors (image_id, skin_color_type)
            VALUES ({image_id}, '{features['skin_color']}')
            """
            if execute_query(conn, query) is None:
                print("Failed to insert skin color")
                conn.close()
                return False
        
        # Insert into emotions table
        if features['emotion'] is not None:
            query = f"""
            INSERT INTO emotions (image_id, emotion_type)
            VALUES ({image_id}, '{features['emotion']}')
            """
            if execute_query(conn, query) is None:
                print("Failed to insert emotion")
                conn.close()
                return False
        
        # Insert into feature_vectors table
        if features['gender_vector'] is not None and features['skin_vector'] is not None and \
           features['emotion_vector'] is not None and features['combined_vector'] is not None:
            gender_vector_json = json.dumps(features['gender_vector'].tolist())
            skin_vector_json = json.dumps(features['skin_vector'].tolist())
            emotion_vector_json = json.dumps(features['emotion_vector'].tolist())
            combined_vector_json = json.dumps(features['combined_vector'].tolist())
            
            query = f"""
            INSERT INTO feature_vectors (
                image_id, gender_vector, skin_vector, 
                emotion_vector, combined_vector
            )
            VALUES (
                {image_id}, 
                '{gender_vector_json}', 
                '{skin_vector_json}', 
                '{emotion_vector_json}', 
                '{combined_vector_json}'
            )
            """
            if execute_query(conn, query) is None:
                print("Failed to insert feature vectors")
                conn.close()
                return False
        
        # Commit changes
        conn.commit()
        
        # Close connection
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"Error adding image to database: {e}")
        # Attempt to close connection in case of error
        if 'conn' in locals() and conn:
            try:
                conn.close()
            except:
                pass
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
        # Tìm image_id dựa trên đường dẫn
        find_image_query = f"SELECT id FROM images WHERE image_path = '{image_path}'"
        
        result = execute_query(connection, find_image_query)
        
        if not result or len(result) == 0:
            print(f"Image {image_path} not found in database")
            connection.close()
            return False
        
        image_id = result[0][0]
        
        # Xóa các record liên quan từ các bảng con
        tables = ["feature_vectors", "face_encodings", "genders", "skin_colors", "emotions"]
        for table in tables:
            delete_query = f"DELETE FROM {table} WHERE image_id = {image_id}"
            delete_result = execute_query(connection, delete_query)
            if delete_result is None:
                print(f"Failed to delete from {table}")
                connection.rollback()
                connection.close()
                return False
        
        # Xóa record từ bảng images
        delete_image_query = f"DELETE FROM images WHERE id = {image_id}"
        delete_result = execute_query(connection, delete_image_query)
        if delete_result is None:
            print("Failed to delete from images table")
            connection.rollback()
            connection.close()
            return False
        
        connection.commit()
        connection.close()
        return True
        
    except Error as e:
        print(f"Error deleting image from database: {e}")
        try:
            connection.rollback()
        except:
            pass
        return False
    finally:
        # Đảm bảo đóng kết nối
        if connection:
            try:
                if connection.is_connected():
                    connection.close()
            except:
                pass

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
            result = execute_query(conn, query)
            if result is None:
                print(f"Failed to delete data from {table}")
                conn.rollback()
                conn.close()
                return False
        
        # Reset auto-increment values
        for table in tables:
            query = f"ALTER TABLE {table} AUTO_INCREMENT = 1"
            result = execute_query(conn, query)
            if result is None:
                print(f"Failed to reset auto-increment for {table}")
                conn.rollback()
                conn.close()
                return False
        
        # Commit changes
        conn.commit()
        
        # Close connection
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"Error clearing database: {e}")
        # Attempt to close connection in case of error
        if 'conn' in locals() and conn:
            try:
                conn.rollback()
                conn.close()
            except:
                pass
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
            conn.close()
            return True, 0
        
        # Process each image
        count = 0
        for image_path in image_files:
            print(f"Đang xử lý {image_path}...")
            
            # Extract features
            features = extract_features(image_path=image_path)
            
            # Skip if no face found
            if not features or not features.get('face_found', False):
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
        # Đảm bảo đóng kết nối
        if 'conn' in locals() and conn:
            try:
                conn.close()
            except:
                pass
        return False, 0


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
        
        # Add filter conditions if provided
        if filters:
            if 'gender' in filters and filters['gender']:
                where_clauses.append(f"g.gender_type = '{filters['gender']}'")
            
            if 'skin_color' in filters and filters['skin_color']:
                where_clauses.append(f"s.skin_color_type = '{filters['skin_color']}'")
            
            if 'emotion' in filters and filters['emotion']:
                where_clauses.append(f"e.emotion_type = '{filters['emotion']}'")
        
        # Add WHERE clause if there are filters
        if where_clauses:
            base_query += " WHERE " + " AND ".join(where_clauses)
        
        # Execute query
        print(f"Executing query: {base_query}")
        
        rows = execute_query(conn, base_query)
        
        # Nếu rows là None hoặc là danh sách trống, trả về danh sách trống
        if rows is None:
            print("Error executing query.")
            conn.close()
            return []
        
        if len(rows) == 0:
            print("Query returned no results.")
            conn.close()
            return []
        
        # Tạo danh sách các kết quả dưới dạng dictionary
        results = []
        for row in rows:
            result = {
                'id': row[0],
                'image_path': row[1],
                'gender_type': row[2],
                'skin_color_type': row[3],
                'emotion_type': row[4],
                'combined_vector': row[5]
            }
            results.append(result)
        
        print(f"Query executed. Found {len(results)} records.")
        
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
        conn.close()
        
        return similar_faces
        
    except Exception as e:
        print(f"Error finding similar faces: {e}")
        import traceback
        traceback.print_exc()
        # Attempt to close connection in case of error
        if 'conn' in locals() and conn:
            try:
                conn.close()
            except:
                pass
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
        
        # Get unique genders
        gender_rows = execute_query(conn, "SELECT DISTINCT gender_type FROM genders")
        genders = [row[0] for row in gender_rows] if gender_rows and gender_rows is not None and len(gender_rows) > 0 else []
        
        # Get unique skin colors
        skin_rows = execute_query(conn, "SELECT DISTINCT skin_color_type FROM skin_colors")
        skin_colors = [row[0] for row in skin_rows] if skin_rows and skin_rows is not None and len(skin_rows) > 0 else []
        
        # Get unique emotions
        emotion_rows = execute_query(conn, "SELECT DISTINCT emotion_type FROM emotions")
        emotions = [row[0] for row in emotion_rows] if emotion_rows and emotion_rows is not None and len(emotion_rows) > 0 else []
        
        # Close connection
        conn.close()
        
        return {
            'genders': genders,
            'skin_colors': skin_colors,
            'emotions': emotions
        }
        
    except Exception as e:
        print(f"Error getting all features: {e}")
        # Attempt to close connection in case of error
        if 'conn' in locals() and conn:
            try:
                conn.close()
            except:
                pass
        return {
            'genders': [],
            'skin_colors': [],
            'emotions': []
        }