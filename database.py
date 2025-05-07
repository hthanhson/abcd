import os
import pickle
import face_recognition
import numpy as np
from feature_extraction import extract_features, copy_image_to_category_folders
from age_estimation import calibrate_age_estimation

def initialize_database():
    """Initialize or load the feature database"""
    features_file = 'features.pkl'
    
    if os.path.exists(features_file):
        with open(features_file, 'rb') as f:
            features_db = pickle.load(f)
    else:
        features_db = {
            'encodings': [],
            'image_paths': [],
            'emotions': [],
            'ages': [],
            'age_groups': [],
            'skin_colors': []
        }
    
    return features_db

def build_database(data_folder, organized_data_folder, features_file='features.pkl'):
    """Process all images in data folder and build feature database"""
    # Calibrate age estimation if possible
    print("Calibrating age estimation...")
    calibrate_age_estimation(data_folder)
    
    # Get all image files
    image_files = [f for f in os.listdir(data_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Initialize features database
    features_db = {
        'encodings': [],
        'image_paths': [],
        'emotions': [],
        'ages': [],
        'age_groups': [],
        'skin_colors': []
    }
    
    total = len(image_files)
    print(f"Processing {total} images...")
    
    for i, img_file in enumerate(image_files):
        print(f"Processing image {i+1}/{total}: {img_file}")
        img_path = os.path.join(data_folder, img_file)
        encoding, emotion, age, age_group, skin_color = extract_features(img_path)
        
        if encoding is not None:
            features_db['encodings'].append(encoding)
            features_db['image_paths'].append(img_path)
            features_db['emotions'].append(emotion)
            features_db['ages'].append(age)
            features_db['age_groups'].append(age_group)
            features_db['skin_colors'].append(skin_color)
            
            # Organize images by categories
            copy_image_to_category_folders(img_path, emotion, age_group, skin_color, organized_data_folder)
    
    # Save the features database
    with open(features_file, 'wb') as f:
        pickle.dump(features_db, f)
    
    print(f"Database built with {len(features_db['encodings'])} faces")
    return len(features_db['encodings'])

def find_similar_faces(query_encoding, features_db, top_n=3):
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
            'age_group': features_db['age_groups'][i],
            'skin_color': features_db['skin_colors'][i]
        })
    
    return result

def filter_database(features_db, emotion=None, min_age=0, max_age=100, age_group=None, skin_color=None):
    """Filter database by various criteria"""
    results = []
    
    for i, img_path in enumerate(features_db['image_paths']):
        matches_emotion = not emotion or features_db['emotions'][i] == emotion
        matches_age = min_age <= features_db['ages'][i] <= max_age
        matches_age_group = not age_group or features_db['age_groups'][i] == age_group
        matches_skin_color = not skin_color or features_db['skin_colors'][i] == skin_color
        
        if matches_emotion and matches_age and matches_age_group and matches_skin_color:
            results.append({
                'image_path': img_path,
                'emotion': features_db['emotions'][i],
                'age': features_db['ages'][i],
                'age_group': features_db['age_groups'][i],
                'skin_color': features_db['skin_colors'][i]
            })
    
    return results 