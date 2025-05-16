from flask import Flask, render_template, request, jsonify, url_for, send_from_directory, redirect
import os
import base64
import cv2
import numpy as np
import json
from werkzeug.utils import secure_filename
from datetime import datetime
from feature_extraction import extract_features
from database import (
    build_database, clear_database, find_similar_faces, get_all_features
)
import utils

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}

# Đảm bảo thư mục uploads tồn tại
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if a file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload for search or database building"""
    try:
        if 'file' not in request.files:
            app.logger.error("Upload error: No file part in request")
            return jsonify({'success': False, 'error': 'Không có file trong yêu cầu'})
        
        file = request.files['file']
        
        if file.filename == '':
            app.logger.error("Upload error: Empty filename")
            return jsonify({'success': False, 'error': 'Không có file được chọn'})
        
        if not allowed_file(file.filename):
            app.logger.error(f"Upload error: Invalid file type - {file.filename}")
            return jsonify({'success': False, 'error': 'Loại file không được hỗ trợ'})
        
        # Kiểm tra kích thước file (giới hạn 10MB)
        file_content = file.read()
        file_size = len(file_content)
        max_size = 10 * 1024 * 1024  # 10MB
        
        if file_size > max_size:
            app.logger.error(f"Upload error: File too large - {file_size} bytes")
            return jsonify({'success': False, 'error': 'File quá lớn. Vui lòng chọn file nhỏ hơn 10MB'})
        
        # Đặt con trỏ về đầu file để sử dụng lại
        file.seek(0)
        
        # Create a unique filename
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        
        # Make sure the uploads directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        app.logger.info(f"File successfully uploaded: {file_path}")
        return jsonify({'success': True, 'file_path': file_path})
    
    except Exception as e:
        app.logger.error(f"Upload error: {str(e)}")
        return jsonify({'success': False, 'error': f'Lỗi khi xử lý file: {str(e)}'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Phục vụ file đã upload"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/build-database', methods=['POST'])
def build_db():
    """Xây dựng database từ thư mục ảnh"""
    try:
        # Kiểm tra xem dữ liệu được gửi dưới dạng nào
        if request.is_json:
            # Nếu là JSON
            data = request.get_json()
            folder_path = data.get('folder_path', '')
        else:
            # Nếu là form data
            folder_path = request.form.get('folder_path', '')
        
        # Xử lý đường dẫn
        folder_path = folder_path.strip()
        
        # Kiểm tra đường dẫn
        if not folder_path:
            return jsonify({'success': False, 'error': 'Đường dẫn thư mục không được để trống'})
            
        # Mở rộng đường dẫn người dùng (hỗ trợ ~, biến môi trường, etc.)
        folder_path = os.path.expanduser(folder_path)
        folder_path = os.path.expandvars(folder_path)
        folder_path = os.path.abspath(folder_path)
        
        # Kiểm tra xem thư mục có tồn tại không
        if not os.path.exists(folder_path):
            return jsonify({'success': False, 'error': f'Thư mục không tồn tại: {folder_path}'})
            
        if not os.path.isdir(folder_path):
            return jsonify({'success': False, 'error': f'{folder_path} không phải là thư mục'})
        
        print(f"Xây dựng CSDL từ thư mục: {folder_path}")
        success, count = build_database(folder_path)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Đã thêm {count} ảnh vào cơ sở dữ liệu'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Không thể xây dựng cơ sở dữ liệu'
            })
    except Exception as e:
        print(f"Error in build_db: {e}")
        return jsonify({
            'success': False,
            'error': f'Lỗi xử lý: {str(e)}'
        })

@app.route('/clear-database', methods=['POST'])
def clear_db():
    """Xóa tất cả dữ liệu từ database"""
    try:
        result = clear_database()
        
        if result:
            return jsonify({
                'success': True,
                'message': 'Đã xóa cơ sở dữ liệu thành công'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Không thể xóa cơ sở dữ liệu'
            })
    except Exception as e:
        print(f"Error in clear_db: {e}")
        return jsonify({
            'success': False,
            'error': f'Lỗi xử lý: {str(e)}'
        })

def normalize_image_path(image_path):
    """
    Normalize an image path to make it web-friendly
    
    Args:
        image_path: The raw image path from database
        
    Returns:
        str: A web-friendly path that can be accessed via HTTP
    """
    if not image_path:
        return "/static/images/error.png"
        
    # Replace backslashes with forward slashes
    path = image_path.replace('\\', '/')
    
    # Handle absolute paths with drive letters (Windows-style paths)
    if ':' in path:
        # Extract just the filename from malformed data_test paths
        if 'data_test' in path:
            # Handle malformed paths like 'E:/hcsdldptNew folderabcddata_test7_0_0_20170110215648859.jpg'
            if 'folderabcddata_test' in path:
                parts = path.split('data_test')
                if len(parts) > 1 and parts[1]:
                    # Just extract the filename part
                    return f"/data_test/{parts[1]}"
            
            # Normal case - just use basename
            filename = os.path.basename(path)
            return f"/data_test/{filename}"
            
        # For uploaded files
        elif 'uploads' in path:
            filename = os.path.basename(path)
            return f"/uploads/{filename}"
    
    # Handle paths that already start with data_test but not with /
    if 'data_test/' in path and not path.startswith('/'):
        return f"/{path}"
        
    # Return the path as is if it's already web-friendly
    return path

@app.route('/search', methods=['POST'])
def search():
    """Search for similar faces"""
    try:
        # Get file path from request
        file_path = request.form.get('file_path')
        print(f"=== SEARCH FUNCTION CALLED ===")
        print(f"File path received: {file_path}")
        
        # Extract features from the input image
        print(f"Extracting features from image...")
        features = extract_features(image_path=file_path)
        
        if not features['face_found']:
            print(f"No face found in the uploaded image")
            return jsonify({
                'success': False,
                'error': 'No face found in the uploaded image'
            })
        
        print(f"Face found. Gender: {features['gender']}, Skin color: {features['skin_color']}, Emotion: {features['emotion']}")
        
        # Find similar faces - không sử dụng bộ lọc
        print(f"Finding similar faces in database...")
        similar_faces = find_similar_faces(features, top_n=3, filters=None)
        
        if not similar_faces:
            print(f"No similar faces found in the database")
            return jsonify({
                'success': False,
                'error': 'No similar faces found in the database'
            })
        
        print(f"Found {len(similar_faces)} similar faces")
        
        # Prepare result for the frontend
        results = []
        for i, face in enumerate(similar_faces):
            # Normalize image path to make it web-friendly
            image_path = face['image_path']
            relative_path = normalize_image_path(image_path)
            
            result = {
                'image_path': relative_path,
                'gender': utils.translate_gender(face['gender_type']),
                'skin_color': utils.translate_skin_color(face['skin_color_type']),
                'emotion': utils.translate_emotion(face['emotion_type']),
                'rank': face['rank']
            }
            print(f"Result #{i+1}: {result}")
            results.append(result)
        
        # Add query image information - normalize the path
        query_relative_path = normalize_image_path(file_path)
            
        query_info = {
            'image_path': query_relative_path,
            'gender': utils.translate_gender(features['gender']),
            'skin_color': utils.translate_skin_color(features['skin_color']),
            'emotion': utils.translate_emotion(features['emotion'])
        }
        
        print(f"Query info: {query_info}")
        print(f"Returning results to frontend...")
        
        response_data = {
            'success': True,
            'query': query_info,
            'results': results
        }
        print(f"Response data: {json.dumps(response_data, indent=2)}")
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error in search: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error processing search: {str(e)}'
        })

@app.route('/data_test/<path:filename>')
def serve_data_test(filename):
    """Phục vụ file từ thư mục data_test"""
    try:
        print(f"Original requested filename: {filename}")
        
        # Fix malformed paths that might be coming through
        if 'folderabcddata_test' in filename:
            parts = filename.split('data_test')
            if len(parts) > 1 and parts[1]:
                filename = parts[1]
                print(f"Extracted filename from malformed path: {filename}")
                
        # Normalize filename by removing any path information and getting just the basename
        # This prevents directory traversal attacks and handles incorrect path formats
        filename = os.path.basename(filename.replace('\\', '/'))
        
        # Clean any URL-encoded components that might remain
        if '%20' in filename:
            filename = filename.replace('%20', ' ')
        
        print(f"Normalized filename: {filename}")
        
        # Absolute path to data_test directory
        data_test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_test')
        
        print(f"Looking in directory: {data_test_dir}")
        
        # List available files in that directory for debugging
        files = os.listdir(data_test_dir)
        if filename in files:
            print(f"File {filename} found in directory listing")
        else:
            print(f"File {filename} NOT found in directory. Available files: {', '.join(files[:5])}...")
        
        # Ensure the file exists
        full_path = os.path.join(data_test_dir, filename)
        if os.path.exists(full_path):
            print(f"File exists: {full_path}")
            return send_from_directory(data_test_dir, filename)
        else:
            print(f"File does not exist: {full_path}")
            return send_from_directory('static/images', 'error.png')
    
    except Exception as e:
        print(f"Error serving data_test file: {e}")
        return send_from_directory('static/images', 'error.png')

# Đảm bảo thư mục data_test tồn tại
os.makedirs('data_test', exist_ok=True)

@app.route('/get-features', methods=['GET'])
def get_features_api():
    """Lấy danh sách các đặc trưng duy nhất từ cơ sở dữ liệu"""
    try:
        features = get_all_features()
        
        # Chuyển đổi sang tiếng Việt
        translated_features = {
            'genders': [utils.translate_gender(gender) for gender in features['genders']],
            'skin_colors': [utils.translate_skin_color(skin) for skin in features['skin_colors']],
            'emotions': [utils.translate_emotion(emotion) for emotion in features['emotions']]
        }
        
        return jsonify({
            'success': True,
            'features': translated_features
        })
    except Exception as e:
        print(f"Error in get_features_api: {e}")
        return jsonify({
            'success': False,
            'error': f'Lỗi khi lấy đặc trưng: {str(e)}'
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)