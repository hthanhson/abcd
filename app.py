from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import os
import base64
import uuid
import sys
from werkzeug.utils import secure_filename
from feature_extraction import extract_features
from database import initialize_database, build_database, find_similar_faces, clear_database, db_manager
import mysql_setup  # Import module thiết lập MySQL

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATA_FOLDER'] = 'data_test'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Khởi tạo MySQL trước khi chạy ứng dụng
mysql_setup.create_database()

# Initialize features database (now using MySQL Connector)
features_db = initialize_database()

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/build-database', methods=['POST'])
def build_db_route():
    """Build or rebuild the feature database"""
    count = build_database(
        app.config['DATA_FOLDER']
    )
    
    # Reload the database after building
    global features_db
    features_db = initialize_database()
    
    return jsonify({
        'status': 'success',
        'message': f'Database built with {count} faces'
    })

@app.route('/clear-database', methods=['POST'])
def clear_db_route():
    """Xóa toàn bộ dữ liệu trong database"""
    success = clear_database()
    
    # Khởi tạo lại connection
    global features_db
    features_db = initialize_database()
    
    if success:
        return jsonify({
            'status': 'success',
            'message': 'Database has been cleared successfully'
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Failed to clear database'
        })

@app.route('/search', methods=['POST'])
def search():
    """Search for similar faces using an uploaded image"""
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
        similar_faces = find_similar_faces(
            encoding,
            top_n=3,
            query_emotion=emotion,
            query_age=age,
            query_age_group=age_group,
            query_skin_color=skin_color
        )
        
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

# Routes for serving static files - cần thiết để ứng dụng hoạt động
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/data_test/<path:filename>')
def serve_data_test(filename):
    return send_from_directory(app.config['DATA_FOLDER'], filename)

@app.route('/features')
def show_features():
    """Hiển thị tất cả các đặc trưng từ database"""
    try:
        # Lấy dữ liệu từ database
        results = db_manager.execute_query(
            """
            SELECT 
                i.image_id,
                i.image_path,
                e.emotion_type AS emotion,
                a.estimated_age AS age,
                s.skin_color
            FROM images i
            LEFT JOIN emotions e ON i.image_id = e.image_id
            LEFT JOIN ages a ON i.image_id = a.image_id
            LEFT JOIN skin_colors s ON i.image_id = s.image_id
            LIMIT 100
            """,
            fetch_all=True
        )
        
        # Xử lý đường dẫn ảnh để hiển thị
        for result in results:
            # Lấy đường dẫn tương đối từ đường dẫn đầy đủ
            image_path = result['image_path']
            if image_path.startswith(app.config['DATA_FOLDER']):
                result['image_url'] = '/' + image_path
            else:
                result['image_url'] = '/data_test/' + os.path.basename(image_path)
        
        # Render template
        return render_template('features.html', features=results)
    except Exception as e:
        print(f"Error retrieving features: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error retrieving features: {str(e)}'
        })

# Compatibility route for backward compatibility
@app.route('/images/<path:filename>')
def get_image(filename):
    # Determine base directory based on path
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
    return send_from_directory(os.path.dirname(filename), os.path.basename(filename))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'build_db':
            # Chạy build database
            build_database(
                app.config['DATA_FOLDER']
            )
        elif sys.argv[1] == '--port' and len(sys.argv) > 2:
            # Chạy ứng dụng với port tùy chỉnh
            port = int(sys.argv[2])
            app.run(debug=True, port=port)
    else:
        app.run(debug=True) 