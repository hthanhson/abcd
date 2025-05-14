import mysql.connector
import db_config

def setup_database():
    """
    Set up the MySQL database with all necessary tables
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Connect to MySQL server (without specifying database)
        conn = mysql.connector.connect(
            host=db_config.DB_HOST,
            user=db_config.DB_USER,
            password=db_config.DB_PASSWORD
        )
        
        cursor = conn.cursor()
        
        # Create database if it doesn't exist
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_config.DB_NAME}")
        
        # Use the database
        cursor.execute(f"USE {db_config.DB_NAME}")
        
        # Create 'images' table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INT AUTO_INCREMENT PRIMARY KEY,
            image_path VARCHAR(255) NOT NULL,
            added_date DATETIME NOT NULL
        )
        """)
        
        # Create 'face_encodings' table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_encodings (
            id INT AUTO_INCREMENT PRIMARY KEY,
            image_id INT NOT NULL,
            face_encoding JSON NOT NULL,
            FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
        )
        """)
        
        # Create 'genders' table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS genders (
            id INT AUTO_INCREMENT PRIMARY KEY,
            image_id INT NOT NULL,
            gender_type ENUM('Man', 'Woman', 'Unknown') NOT NULL,
            confidence_value FLOAT NOT NULL,
            FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
        )
        """)
        
        # Create 'skin_colors' table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS skin_colors (
            id INT AUTO_INCREMENT PRIMARY KEY,
            image_id INT NOT NULL,
            skin_color_type ENUM('White', 'Black', 'Yellow', 'Unknown') NOT NULL,
            FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
        )
        """)
        
        # Create 'emotions' table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS emotions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            image_id INT NOT NULL,
            emotion_type ENUM('Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral', 'Unknown') NOT NULL,
            FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
        )
        """)
        
        # Create 'feature_vectors' table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS feature_vectors (
            id INT AUTO_INCREMENT PRIMARY KEY,
            image_id INT NOT NULL,
            gender_vector JSON NOT NULL,
            skin_vector JSON NOT NULL,
            emotion_vector JSON NOT NULL,
            combined_vector JSON NOT NULL,
            FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
        )
        """)
        
        # Commit changes
        conn.commit()
        
        print("Database setup completed successfully!")
        print(f"Database name: {db_config.DB_NAME}")
        print(f"Tables created: images, face_encodings, genders, skin_colors, emotions, feature_vectors")
        
        # Close connection
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"Error setting up database: {e}")
        return False

if __name__ == "__main__":
    setup_database()