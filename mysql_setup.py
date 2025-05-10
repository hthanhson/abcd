"""
Script để tạo cơ sở dữ liệu MySQL và các bảng cần thiết
"""

import mysql.connector
from mysql.connector import Error
from db_config import MYSQL_CONFIG

def create_database():
    """Tạo cơ sở dữ liệu nếu chưa tồn tại"""
    try:
        # Kết nối tới MySQL mà không chỉ định database
        conn = mysql.connector.connect(
            host=MYSQL_CONFIG['host'],
            user=MYSQL_CONFIG['user'],
            password=MYSQL_CONFIG['password']
        )
        
        cursor = conn.cursor()
        
        # Tạo database
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_CONFIG['database']}")
        print(f"Database {MYSQL_CONFIG['database']} created successfully")
        
        cursor.close()
        conn.close()
        
        # Sau khi tạo database, tạo các bảng
        create_tables()
        
    except Error as e:
        print(f"Error creating database: {e}")

def create_tables():
    """Tạo các bảng cần thiết trong cơ sở dữ liệu"""
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        
        # Bảng ảnh (bảng chính)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            image_id INT AUTO_INCREMENT PRIMARY KEY,
            image_name VARCHAR(255) NOT NULL,
            image_path VARCHAR(512) NOT NULL,
            upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY (image_path)
        )
        """)
        
        # Bảng đặc trưng khuôn mặt (encodings)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_encodings (
            encoding_id INT AUTO_INCREMENT PRIMARY KEY,
            image_id INT NOT NULL,
            encoding_data TEXT NOT NULL,
            FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE,
            INDEX (image_id)
        )
        """)
        
        # Bảng cảm xúc
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS emotions (
            emotion_id INT AUTO_INCREMENT PRIMARY KEY,
            image_id INT NOT NULL,
            emotion_type ENUM('happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral') NOT NULL,
            confidence_value FLOAT NOT NULL DEFAULT 0,
            FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE,
            INDEX (image_id),
            INDEX (emotion_type)
        )
        """)
        
        # Bảng tuổi
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ages (
            age_id INT AUTO_INCREMENT PRIMARY KEY,
            image_id INT NOT NULL,
            estimated_age FLOAT NOT NULL,
            confidence_value FLOAT NOT NULL DEFAULT 0,
            FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE,
            INDEX (image_id)
        )
        """)
        
        # Bảng nhóm tuổi
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS age_groups (
            age_group_id INT AUTO_INCREMENT PRIMARY KEY,
            image_id INT NOT NULL,
            age_group ENUM('child', 'teen', 'adult', 'senior') NOT NULL,
            confidence_value FLOAT NOT NULL DEFAULT 1.0,
            FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE,
            INDEX (image_id),
            INDEX (age_group)
        )
        """)
        
        # Bảng màu da
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS skin_colors (
            skin_color_id INT AUTO_INCREMENT PRIMARY KEY,
            image_id INT NOT NULL,
            skin_color ENUM('White', 'Black', 'Yellow', 'unknown') NOT NULL,
            confidence_value FLOAT NOT NULL DEFAULT 1.0,
            FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE,
            INDEX (image_id),
            INDEX (skin_color)
        )
        """)
        
        print("All tables created successfully")
        
        cursor.close()
        conn.close()
        
    except Error as e:
        print(f"Error creating tables: {e}")

if __name__ == "__main__":
    print("Setting up MySQL database...")
    create_database()
    print("Setup complete.") 