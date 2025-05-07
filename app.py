from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import os
import base64
import uuid
import sys
from werkzeug.utils import secure_filename
from feature_extraction import extract_features
from database import initialize_database, build_database, find_similar_faces, filter_database

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATA_FOLDER'] = 'data_test'
app.config['ORGANIZED_DATA_FOLDER'] = 'data'
app.config['FEATURES_FILE'] = 'features.pkl'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ORGANIZED_DATA_FOLDER'], exist_ok=True)

# Initialize features database
features_db = initialize_database()

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/build-database', methods=['POST'])
def build_db_route():
    """Build or rebuild the feature database"""
    count = build_database(
        app.config['DATA_FOLDER'], 
        app.config['ORGANIZED_DATA_FOLDER'],
        app.config['FEATURES_FILE']
    )
    
    # Reload the database after building
    global features_db
    features_db = initialize_database()
    
    return jsonify({
        'status': 'success',
        'message': f'Database built with {count} faces'
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
        similar_faces = find_similar_faces(encoding, features_db)
        
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
    """Filter images by various criteria"""
    emotion = request.form.get('emotion', '')
    min_age = int(request.form.get('min_age', 0))
    max_age = int(request.form.get('max_age', 100))
    age_group = request.form.get('age_group', '')
    skin_color = request.form.get('skin_color', '')
    
    # Filter database by criteria
    results = filter_database(
        features_db, 
        emotion=emotion, 
        min_age=min_age, 
        max_age=max_age, 
        age_group=age_group, 
        skin_color=skin_color
    )
    
    return jsonify({
        'status': 'success',
        'results': results
    })

@app.route('/api/category/<category_type>/<category_name>', methods=['GET'])
def get_category_images(category_type, category_name):
    """Return a list of images from the specified category folder"""
    if category_type not in ['emotions', 'age', 'skin']:
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
                'skin_color': category_name if category_type == 'skin' else 'unknown'
            })
    
    return jsonify({
        'status': 'success',
        'category_type': category_type,
        'category_name': category_name,
        'count': len(images),
        'images': images
    })

# Routes for serving static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory(app.config['ORGANIZED_DATA_FOLDER'], filename)

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/data_test/<path:filename>')
def serve_data_test(filename):
    return send_from_directory(app.config['DATA_FOLDER'], filename)

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
    elif filename.startswith(app.config['ORGANIZED_DATA_FOLDER']):
        return send_from_directory(
            app.config['ORGANIZED_DATA_FOLDER'],
            filename.replace(app.config['ORGANIZED_DATA_FOLDER'] + '/', '')
        )
    return send_from_directory(os.path.dirname(filename), os.path.basename(filename))

if __name__ == '__main__':
    # Check if we should just build the database
    if len(sys.argv) > 1 and sys.argv[1] == 'build_db':
        build_database(
            app.config['DATA_FOLDER'], 
            app.config['ORGANIZED_DATA_FOLDER'],
            app.config['FEATURES_FILE']
        )
    else:
        app.run(debug=True) 