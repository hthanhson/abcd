#!/usr/bin/env python3
"""
Benchmark script for the Face Recognition System
This script evaluates the performance of the face recognition system.
"""

import os
import time
import pickle
import face_recognition
import numpy as np
from deepface import DeepFace
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random
import cv2

# Configuration
DATA_FOLDER = 'data_test'
FEATURES_FILE = 'features.pkl'
BENCHMARK_FOLDER = 'benchmark_results'
NUM_TEST_IMAGES = 10  # Number of images to use for testing

def load_or_build_database():
    """Load or build the features database"""
    if os.path.exists(FEATURES_FILE):
        print(f"Loading existing features database from {FEATURES_FILE}")
        with open(FEATURES_FILE, 'rb') as f:
            features_db = pickle.load(f)
        return features_db
    else:
        print("No features database found. Run app.py first to build the database.")
        return None

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
        analysis = DeepFace.analyze(image_path, actions=['emotion', 'age'], enforce_detection=False)
        emotion = analysis[0]['dominant_emotion']
        age = analysis[0]['age']
        
        # Extract skin color
        top, right, bottom, left = face_locations[0]
        face_image = image[top:bottom, left:right]
        hsv_face = cv2.cvtColor(face_image, cv2.COLOR_RGB2HSV)
        skin_color = np.mean(hsv_face[:, :, 0])
        
        return encoding, emotion, age, skin_color
    except Exception as e:
        print(f"Error analyzing face: {e}")
        return encoding, "unknown", 0, 0

def get_prefix_from_filename(filename):
    """Extract the prefix (e.g. 16_1_0_) from a filename"""
    # Assuming format like 16_1_0_20170109214419099.jpg
    parts = filename.split('_')
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}_{parts[2]}_"
    return ""

def benchmark_accuracy(features_db):
    """Benchmark the accuracy of the face recognition system"""
    os.makedirs(BENCHMARK_FOLDER, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Exclude images already in the database
    test_candidates = []
    db_filenames = [os.path.basename(path) for path in features_db['image_paths']]
    
    for img in image_files:
        if img not in db_filenames:
            test_candidates.append(img)
    
    # If not enough test candidates, use some from the database
    if len(test_candidates) < NUM_TEST_IMAGES:
        additional = NUM_TEST_IMAGES - len(test_candidates)
        test_candidates.extend(random.sample(db_filenames, min(additional, len(db_filenames))))
    
    # Randomly select test images
    test_images = random.sample(test_candidates, min(NUM_TEST_IMAGES, len(test_candidates)))
    
    # Tracking metrics
    correct_matches = 0
    total_tests = 0
    emotion_accuracy = 0
    age_differences = []
    processing_times = []
    
    # Prepare for confusion matrix
    actual_classes = []
    predicted_classes = []
    
    # Test each image
    for test_img in test_images:
        test_img_path = os.path.join(DATA_FOLDER, test_img)
        test_prefix = get_prefix_from_filename(test_img)
        
        # Start timer
        start_time = time.time()
        
        # Extract features
        encoding, emotion, age, skin_color = extract_features(test_img_path)
        
        if encoding is None:
            print(f"Skipping {test_img_path} - no face detected")
            continue
        
        # Calculate face distances
        face_distances = face_recognition.face_distance(features_db['encodings'], encoding)
        
        # Sort and get top match
        best_match_idx = np.argmin(face_distances)
        best_match_path = features_db['image_paths'][best_match_idx]
        best_match_filename = os.path.basename(best_match_path)
        best_match_prefix = get_prefix_from_filename(best_match_filename)
        
        # Get all top 3 matches for visualization
        top_indices = np.argsort(face_distances)[:3]
        top_matches = []
        for i in top_indices:
            similarity = 1 - face_distances[i]
            top_matches.append({
                'image_path': features_db['image_paths'][i],
                'similarity': float(similarity),
                'emotion': features_db['emotions'][i],
                'age': features_db['ages'][i]
            })
        
        # End timer
        end_time = time.time()
        processing_time = end_time - start_time
        processing_times.append(processing_time)
        
        # Check if match has the same prefix (same person)
        is_correct = (best_match_prefix == test_prefix)
        
        # For confusion matrix
        actual_classes.append(test_prefix)
        predicted_classes.append(best_match_prefix)
        
        # Check emotion accuracy
        db_emotion = features_db['emotions'][best_match_idx]
        emotion_match = (db_emotion == emotion)
        if emotion_match:
            emotion_accuracy += 1
        
        # Calculate age difference
        db_age = features_db['ages'][best_match_idx]
        age_diff = abs(db_age - age)
        age_differences.append(age_diff)
        
        # Record results
        total_tests += 1
        if is_correct:
            correct_matches += 1
        
        # Create visualization
        visualize_benchmark_result(test_img_path, top_matches, is_correct, emotion_match, age_diff, processing_time)
    
    # Calculate final metrics
    accuracy = correct_matches / total_tests if total_tests > 0 else 0
    emotion_acc = emotion_accuracy / total_tests if total_tests > 0 else 0
    avg_age_diff = sum(age_differences) / len(age_differences) if age_differences else 0
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    
    # Create summary report
    create_report(accuracy, emotion_acc, avg_age_diff, avg_processing_time, total_tests)
    
    # Create confusion matrix visualization
    create_confusion_matrix(actual_classes, predicted_classes)
    
    return accuracy, emotion_acc, avg_age_diff, avg_processing_time

def visualize_benchmark_result(test_img_path, top_matches, is_correct, emotion_match, age_diff, processing_time):
    """Create visualization for a benchmark test"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Query image
    query_img = cv2.imread(test_img_path)
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    axes[0].imshow(query_img)
    axes[0].set_title("Query Image")
    axes[0].axis('off')
    
    # Top matches
    for i, match in enumerate(top_matches):
        img = cv2.imread(match['image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i+1].imshow(img)
        
        # Color title based on match correctness (only for first match)
        if i == 0:
            color = 'green' if is_correct else 'red'
            title = f"Top Match ({match['similarity']*100:.1f}%)\n"
            title += f"Correct: {'✓' if is_correct else '✗'}\n"
            title += f"Emotion: {'✓' if emotion_match else '✗'}\n"
            title += f"Age Diff: {age_diff:.1f} years"
        else:
            color = 'black'
            title = f"Match {i+1} ({match['similarity']*100:.1f}%)"
        
        axes[i+1].set_title(title, color=color)
        axes[i+1].axis('off')
    
    # Add processing time
    fig.text(0.5, 0.01, f"Processing time: {processing_time:.3f} seconds", ha='center')
    
    # Save result
    output_filename = os.path.join(BENCHMARK_FOLDER, f"result_{os.path.basename(test_img_path)}")
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()

def create_report(accuracy, emotion_acc, avg_age_diff, avg_processing_time, total_tests):
    """Create a summary report of the benchmark"""
    report = f"""
    # Face Recognition System Benchmark Report

    ## Summary
    - Total test images: {total_tests}
    - Face recognition accuracy: {accuracy*100:.2f}%
    - Emotion detection accuracy: {emotion_acc*100:.2f}%
    - Average age difference: {avg_age_diff:.2f} years
    - Average processing time: {avg_processing_time:.3f} seconds

    ## Details
    The benchmark tested the system's ability to correctly identify the same person
    in different images, as well as the accuracy of emotion detection and age estimation.
    
    ## Interpretation
    - **Face Recognition**: Measures how often the system correctly identified the same person
    - **Emotion Detection**: Measures if the emotion detected matches the emotion in the database
    - **Age Difference**: Average difference between estimated ages (lower is better)
    - **Processing Time**: Time taken to analyze each image and find matches
    """
    
    with open(os.path.join(BENCHMARK_FOLDER, 'benchmark_report.md'), 'w') as f:
        f.write(report)
    
    # Create a visualization of metrics
    plt.figure(figsize=(10, 6))
    metrics = ['Recognition Accuracy', 'Emotion Accuracy']
    values = [accuracy * 100, emotion_acc * 100]
    
    plt.bar(metrics, values, color=['#6e8efb', '#a777e3'])
    plt.ylabel('Percentage (%)')
    plt.title('Face Recognition System Performance')
    plt.ylim(0, 100)
    
    for i, v in enumerate(values):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center')
    
    plt.savefig(os.path.join(BENCHMARK_FOLDER, 'performance_metrics.png'))
    plt.close()

def create_confusion_matrix(actual, predicted):
    """Create a confusion matrix visualization"""
    # Get unique classes
    classes = sorted(list(set(actual + predicted)))
    
    # Create confusion matrix
    cm = confusion_matrix(actual, predicted, labels=classes)
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, values_format='.2f')
    plt.title('Confusion Matrix (Normalized)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(BENCHMARK_FOLDER, 'confusion_matrix.png'))
    plt.close()

def run_benchmark():
    """Run the benchmark for the face recognition system"""
    print("=" * 50)
    print("Face Recognition System Benchmark")
    print("=" * 50)
    
    # Load database
    features_db = load_or_build_database()
    if not features_db:
        return
    
    print(f"Database contains {len(features_db['encodings'])} faces")
    print("Starting benchmark...")
    
    # Run benchmark
    accuracy, emotion_acc, avg_age_diff, avg_processing_time = benchmark_accuracy(features_db)
    
    # Print results
    print(f"Benchmark completed!")
    print(f"Face recognition accuracy: {accuracy*100:.2f}%")
    print(f"Emotion detection accuracy: {emotion_acc*100:.2f}%")
    print(f"Average age difference: {avg_age_diff:.2f} years")
    print(f"Average processing time: {avg_processing_time:.3f} seconds")
    print(f"Results saved to {BENCHMARK_FOLDER}/ directory")
    print("=" * 50)

if __name__ == "__main__":
    run_benchmark() 