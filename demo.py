#!/usr/bin/env python3
"""
Demo script for the Face Recognition System
This script demonstrates the main functionality of the system with a few test images.
"""

import os
import sys
import face_recognition
import cv2
import numpy as np
from deepface import DeepFace
import matplotlib.pyplot as plt
import pickle

# Configuration
DATA_FOLDER = 'data_test'
FEATURES_FILE = 'features.pkl'
OUTPUT_FOLDER = 'demo_output'

def extract_features(image_path):
    """Extract facial features: encoding, emotion, age, and skin color"""
    print(f"Processing {image_path}...")
    
    # Load image
    image = face_recognition.load_image_file(image_path)
    
    # Find face locations
    face_locations = face_recognition.face_locations(image)
    
    # Check if a face was detected
    if not face_locations:
        print(f"No face detected in {image_path}")
        return None, None, None, None
    
    # Get face encodings
    face_encodings = face_recognition.face_encodings(image, face_locations)
    if not face_encodings:
        print(f"No face encodings found in {image_path}")
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
        
        # Draw rectangle on face
        cv2_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(cv2_image, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Add text with info
        cv2.putText(cv2_image, f"Emotion: {emotion}", (left, top - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(cv2_image, f"Age: {age}", (left, top - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save the annotated image
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(image_path))
        cv2.imwrite(output_path, cv2_image)
        print(f"Saved annotated image to {output_path}")
        
        return encoding, emotion, age, skin_color
    except Exception as e:
        print(f"Error analyzing face: {e}")
        return encoding, "unknown", 0, 0

def build_database():
    """Process all images in data folder and build feature database"""
    data_folder = DATA_FOLDER
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
    
    # Process only a subset for demo
    for i, img_file in enumerate(image_files[:10]):
        print(f"Processing image {i+1}/{10}")
        img_path = os.path.join(data_folder, img_file)
        encoding, emotion, age, skin_color = extract_features(img_path)
        
        if encoding is not None:
            features_db['encodings'].append(encoding)
            features_db['image_paths'].append(img_path)
            features_db['emotions'].append(emotion)
            features_db['ages'].append(age)
            features_db['skin_colors'].append(skin_color)
    
    # Save the features database
    with open(FEATURES_FILE, 'wb') as f:
        pickle.dump(features_db, f)
    
    print(f"Database built with {len(features_db['encodings'])} faces")
    return features_db

def find_similar_faces(query_image, features_db, top_n=3):
    """Find top N similar faces based on facial encoding"""
    encoding, emotion, age, skin_color = extract_features(query_image)
    
    if encoding is None:
        print(f"Could not extract features from {query_image}")
        return []
    
    if not features_db['encodings']:
        print("Database is empty")
        return []
    
    # Calculate face distances
    face_distances = face_recognition.face_distance(features_db['encodings'], encoding)
    
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

def visualize_results(query_image, similar_faces):
    """Visualize the query image and top similar faces"""
    if not similar_faces:
        print("No similar faces found")
        return
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Show query image
    query_img = cv2.imread(query_image)
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    axes[0].imshow(query_img)
    axes[0].set_title("Query Image")
    axes[0].axis('off')
    
    # Show similar faces
    for i, face in enumerate(similar_faces[:3]):
        img = cv2.imread(face['image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i+1].imshow(img)
        similarity_pct = face['similarity'] * 100
        axes[i+1].set_title(f"Match {i+1}: {similarity_pct:.2f}%\nEmotion: {face['emotion']}\nAge: {face['age']}")
        axes[i+1].axis('off')
    
    # Save and show
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_path = os.path.join(OUTPUT_FOLDER, 'similar_faces.png')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved similar faces visualization to {output_path}")

def run_demo():
    """Run a demonstration of the face recognition system"""
    print("=" * 50)
    print("Face Recognition System Demo")
    print("=" * 50)
    
    # Check if features database exists, otherwise build it
    if os.path.exists(FEATURES_FILE):
        print(f"Loading existing features database from {FEATURES_FILE}")
        with open(FEATURES_FILE, 'rb') as f:
            features_db = pickle.load(f)
    else:
        print("Building features database...")
        features_db = build_database()
    
    print(f"Database contains {len(features_db['encodings'])} faces")
    
    # Select a random image for testing
    data_folder = DATA_FOLDER
    image_files = [f for f in os.listdir(data_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("No images found in data folder")
        return
    
    # Choose a random image that was not in the first 10 used for the database
    if len(image_files) > 10:
        query_image = os.path.join(data_folder, image_files[15])
    else:
        query_image = os.path.join(data_folder, image_files[0])
    
    print(f"Using {query_image} as query image")
    
    # Find similar faces
    print("Finding similar faces...")
    similar_faces = find_similar_faces(query_image, features_db)
    
    # Print results
    print("\nResults:")
    for i, face in enumerate(similar_faces):
        print(f"Match {i+1}: {face['image_path']}")
        print(f"  Similarity: {face['similarity'] * 100:.2f}%")
        print(f"  Emotion: {face['emotion']}")
        print(f"  Age: {face['age']}")
        print()
    
    # Visualize
    visualize_results(query_image, similar_faces)
    
    print("=" * 50)
    print("Demo completed! Check the demo_output folder for results.")
    print("=" * 50)

if __name__ == "__main__":
    run_demo() 