from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import os
import shutil
import face_recognition
import numpy as np
import cv2
from werkzeug.utils import secure_filename
from PIL import Image
import pickle
from deepface import DeepFace
import uuid
import sys

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATA_FOLDER'] = 'data_test'
app.config['FEATURES_FILE'] = 'features.pkl'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load or create features dictionary
if os.path.exists(app.config['FEATURES_FILE']):
    with open(app.config['FEATURES_FILE'], 'rb') as f:
        features_db = pickle.load(f)
else:
    features_db = {
        'encodings': [],
        'image_paths': [],
        'emotions': [],
        'ages': [],
        'skin_colors': []
    }

def extract_features(image_path):
    """Extract facial features: encoding, emotion, age, and skin color"""
    # Load image
    image = face_recognition.load_image_file(image_path)
    # Find face locations
    face_locations = face_recognition.face_locations(image)
    
    # Check if a face was detected
    if not face_locations:
        return None, None, None, None
    
    # Get face encodings
    face_encodings = face_recognition.face_encodings(image, face_locations)
    if not face_encodings:
        return None, None, None, None
    
    encoding = face_encodings[0]  # Use first face
    
    try:
        # Get emotion, age using DeepFace
        analysis = DeepFace.analyze(image_path, actions=['emotion', 'age'])
        emotion = analysis[0]['dominant_emotion']
        age = analysis[0]['age']
        
        # Extract skin color (simplified approach - average color in face region)
        top, right, bottom, left = face_locations[0]
        face_image = image[top:bottom, left:right]
        hsv_face = cv2.cvtColor(face_image, cv2.COLOR_RGB2HSV)
        skin_color = np.mean(hsv_face[:, :, 0])  # Hue value average
        
        return encoding, emotion, age, skin_color
    except Exception as e:
        print(f"Error analyzing face: {e}")
        return encoding, "unknown", 0, 0

def build_database():
    """Process all images in data folder and build feature database"""
    global features_db
    
    data_folder = app.config['DATA_FOLDER']
    image_files = [f for f in os.listdir(data_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    features_db = {
        'encodings': [],
        'image_paths': [],
        'emotions': [],
        'ages': [],
        'skin_colors': []
    }
    
    total = len(image_files)
    print(f"Processing {total} images...")
    
    for i, img_file in enumerate(image_files):
        print(f"Processing image {i+1}/{total}: {img_file}")
        img_path = os.path.join(data_folder, img_file)
        encoding, emotion, age, skin_color = extract_features(img_path)
        
        if encoding is not None:
            features_db['encodings'].append(encoding)
            features_db['image_paths'].append(img_path)
            features_db['emotions'].append(emotion)
            features_db['ages'].append(age)
            features_db['skin_colors'].append(skin_color)
    
    # Save the features database
    with open(app.config['FEATURES_FILE'], 'wb') as f:
        pickle.dump(features_db, f)
    
    print(f"Database built with {len(features_db['encodings'])} faces")
    return len(features_db['encodings'])

def find_similar_faces(query_encoding, top_n=3):
    """Find top N similar faces based on facial encoding"""
    if not features_db['encodings']:
        return []
    
    # Calculate face distances
    face_distances = face_recognition.face_distance(features_db['encodings'], query_encoding)
    
    # Sort and get top N matches
    indices = np.argsort(face_distances)[:top_n]
    
    # Result list
    result = []
    for i in indices:
        similarity = 1 - face_distances[i]  # Convert distance to similarity (0-1)
        result.append({
            'image_path': features_db['image_paths'][i],
            'similarity': float(similarity),
            'emotion': features_db['emotions'][i],
            'age': features_db['ages'][i],
            'skin_color': features_db['skin_colors'][i]
        })
    
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/build-database', methods=['POST'])
def build_db_route():
    count = build_database()
    return jsonify({
        'status': 'success',
        'message': f'Database built with {count} faces'
    })

@app.route('/search', methods=['POST'])
def search():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})
    
    # Save the uploaded file
    filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Extract features
    encoding, emotion, age, skin_color = extract_features(file_path)
    
    if encoding is None:
        os.remove(file_path)  # Clean up
        return jsonify({
            'status': 'error',
            'message': 'No face detected in the uploaded image'
        })
    
    # Find similar faces
    similar_faces = find_similar_faces(encoding)
    
    # Prepare result
    result = {
        'status': 'success',
        'query_image': file_path,
        'query_features': {
            'emotion': emotion,
            'age': age,
            'skin_color': skin_color
        },
        'similar_faces': similar_faces
    }
    
    return jsonify(result)

@app.route('/filter', methods=['POST'])
def filter_images():
    emotion = request.form.get('emotion', '')
    min_age = int(request.form.get('min_age', 0))
    max_age = int(request.form.get('max_age', 100))
    
    results = []
    
    for i, img_path in enumerate(features_db['image_paths']):
        if (not emotion or features_db['emotions'][i] == emotion) and \
           min_age <= features_db['ages'][i] <= max_age:
            results.append({
                'image_path': img_path,
                'emotion': features_db['emotions'][i],
                'age': features_db['ages'][i],
                'skin_color': features_db['skin_colors'][i]
            })
    
    return jsonify({
        'status': 'success',
        'results': results
    })

@app.route('/images/<path:filename>')
def get_image(filename):
    return send_from_directory(os.path.dirname(filename), os.path.basename(filename))

if __name__ == '__main__':
    # Check if we should just build the database
    if len(sys.argv) > 1 and sys.argv[1] == 'build_db':
        build_database()
    else:
        app.run(debug=True) 