"""
Fruit Classification from Conveyor Belt Images

This script implements a machine learning workflow to classify fruit images 
captured from a conveyor belt using the 7 invariant (Hu) moments as features.

Deadline: Wednesday, December 11, 2024, at 14:00

Usage:
    python fruit_classifier.py

Requirements:
    - OpenCV (cv2)
    - NumPy
    - scikit-learn
    - Python 3.6+
"""

import os
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import fbeta_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import random
import time
from datetime import datetime

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def load_and_split_data(base_path, fruit_classes, train_size=42, test_size=18):
    """
    Load images for each fruit class and split into training and testing sets.
    
    Args:
        base_path (str): Path to the directory containing fruit image folders
        fruit_classes (list): List of fruit class names
        train_size (int): Number of images per class for training
        test_size (int): Number of images per class for testing
        
    Returns:
        tuple: (training_paths, training_labels, testing_paths, testing_labels)
    """
    # Validation
    if train_size + test_size != 60:
        raise ValueError(f"train_size ({train_size}) + test_size ({test_size}) must equal 60")
    
    training_paths = []
    training_labels = []
    testing_paths = []
    testing_labels = []
    
    for class_idx, fruit in enumerate(fruit_classes):
        class_dir = os.path.join(base_path, fruit)
        # Get all image paths for this fruit
        image_paths = [os.path.join(class_dir, img) for img in os.listdir(class_dir) 
                       if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Verify we have exactly 60 images
        if len(image_paths) != 60:
            raise ValueError(f"Expected 60 images for {fruit}, found {len(image_paths)}")
        
        # Shuffle the images
        random.shuffle(image_paths)
        
        # Split into training and testing
        train_images = image_paths[:train_size]
        test_images = image_paths[train_size:train_size+test_size]
        
        # Add to our lists
        training_paths.extend(train_images)
        training_labels.extend([class_idx] * train_size)
        testing_paths.extend(test_images)
        testing_labels.extend([class_idx] * test_size)
    
    return training_paths, training_labels, testing_paths, testing_labels

def extract_hu_moments(image_path):
    """
    Extract the 7 Hu moments from an image.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Array of 7 Hu moments
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate Moments
    moments = cv2.moments(gray)
    
    # Calculate Hu Moments
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Log transform to handle scale differences
    # Using -1 * sign(h) * log10(abs(h)) for better numerical stability
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    
    return hu_moments

def extract_features_from_paths(image_paths):
    """
    Extract features from a list of image paths.
    
    Args:
        image_paths (list): List of paths to image files
        
    Returns:
        numpy.ndarray: Matrix of features (one row per image)
    """
    features = []
    for path in image_paths:
        try:
            # Extract Hu moments
            hu_moments = extract_hu_moments(path)
            features.append(hu_moments)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            # Add a row of zeros as fallback
            features.append(np.zeros(7))
    
    return np.array(features)

def extract_enhanced_features(image_path):
    """
    Extract enhanced features from an image, including Hu moments and color histograms.
    Used when basic features don't achieve the required performance.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Array of features
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Extract Hu moments (same as before)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    
    # Extract color histograms
    # Calculate histogram in HSV color space (better for color-based classification)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256])
    
    # Normalize histograms
    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    s_hist = cv2.normalize(s_hist, s_hist).flatten()
    v_hist = cv2.normalize(v_hist, v_hist).flatten()
    
    # Combine all features
    all_features = np.concatenate([
        hu_moments,
        h_hist,
        s_hist,
        v_hist
    ])
    
    return all_features

def extract_enhanced_features_from_paths(image_paths):
    """
    Extract enhanced features from a list of image paths.
    
    Args:
        image_paths (list): List of paths to image files
        
    Returns:
        numpy.ndarray: Matrix of features (one row per image)
    """
    features = []
    feature_length = None
    
    for path in image_paths:
        try:
            # Extract enhanced features
            img_features = extract_enhanced_features(path)
            if feature_length is None:
                feature_length = len(img_features)
            features.append(img_features)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            # Add a row of zeros as fallback
            if feature_length is None:
                feature_length = 55  # 7 Hu moments + 16*3 color histogram bins
            features.append(np.zeros(feature_length))
    
    return np.array(features)

def train_classifier(X_train, y_train, classifier_type='svm'):
    """
    Train a classifier on the given features and labels.
    
    Args:
        X_train (numpy.ndarray): Feature matrix for training
        y_train (numpy.ndarray): Label vector for training
        classifier_type (str): Type of classifier to use ('svm' or 'knn')
        
    Returns:
        object: Trained classifier
    """
    # Create a pipeline with scaling
    if classifier_type.lower() == 'svm':
        # SVM classifier with RBF kernel
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='rbf', C=10, gamma='scale'))
        ])
    elif classifier_type.lower() == 'knn':
        # k-NN classifier
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier(n_neighbors=5))
        ])
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")
    
    # Train the classifier
    pipeline.fit(X_train, y_train)
    
    return pipeline

def tune_classifier(X_train, y_train, classifier_type='svm'):
    """
    Tune a classifier using grid search.
    
    Args:
        X_train (numpy.ndarray): Feature matrix for training
        y_train (numpy.ndarray): Label vector for training
        classifier_type (str): Type of classifier to use ('svm' or 'knn')
        
    Returns:
        object: Tuned classifier
    """
    # Create a pipeline with scaling
    if classifier_type.lower() == 'svm':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(probability=True))
        ])
        
        # Parameters to search
        param_grid = {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__gamma': ['scale', 'auto', 0.1, 0.01],
            'classifier__kernel': ['rbf', 'poly']
        }
    elif classifier_type.lower() == 'knn':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier())
        ])
        
        # Parameters to search
        param_grid = {
            'classifier__n_neighbors': [3, 5, 7, 9],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
        }
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")
    
    # Create grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='f1_weighted',
        verbose=1
    )
    
    # Perform grid search
    print(f"Tuning {classifier_type} classifier...")
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_classifier(classifier, X_test, y_test, beta_values=[0.5, 1.0, 2.0]):
    """
    Evaluate the classifier using F-beta scores.
    
    Args:
        classifier (object): Trained classifier
        X_test (numpy.ndarray): Feature matrix for testing
        y_test (numpy.ndarray): True labels for testing
        beta_values (list): List of beta values to calculate F-beta scores
        
    Returns:
        dict: Dictionary of F-beta scores
    """
    # Make predictions
    y_pred = classifier.predict(X_test)
    
    # Calculate F-beta scores
    results = {}
    for beta in beta_values:
        score = fbeta_score(y_test, y_pred, beta=beta, average='weighted')
        results[f'f{beta}'] = score
        print(f"F-{beta} Score: {score:.4f}")
    
    return results

def main():
    """
    Main function to execute the fruit classification workflow.
    
    Project: Fruit Classification from Conveyor Belt Images
    Deadline: Wednesday, December 11, 2024, at 14:00
    """
    start_time = time.time()
    print("Starting fruit classification workflow")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define fruit classes
    fruit_classes = ['avocado', 'eggplant', 'tangerine', 'apple', 'watermelon']
    
    # Set paths (update with your actual path)
    base_path = "./fruit_images"  # Replace with actual path to dataset
    
    # Step 1: Load and split the data
    print("\nStep 1: Loading and splitting data...")
    try:
        train_paths, train_labels, test_paths, test_labels = load_and_split_data(
            base_path, fruit_classes)
        
        print(f"Loaded {len(train_paths)} training images and {len(test_paths)} testing images")
    except Exception as e:
        print(f"Error in data loading: {e}")
        print("Please check that the dataset is structured correctly:")
        print("- Base directory should contain subdirectories for each fruit type")
        print("- Each fruit subdirectory should contain 60 images")
        return
    
    # Step 2: Extract basic features from training images
    print("\nStep 2: Extracting Hu moment features from training images...")
    X_train = extract_features_from_paths(train_paths)
    y_train = np.array(train_labels)
    
    # Step 3: Extract basic features from testing images
    print("\nStep 3: Extracting Hu moment features from testing images...")
    X_test = extract_features_from_paths(test_paths)
    y_test = np.array(test_labels)
    
    # Step 4: Train basic classifier
    print("\nStep 4: Training classifier with Hu moment features...")
    classifier = train_classifier(X_train, y_train, classifier_type='svm')
    
    # Step 5: Evaluate basic classifier
    print("\nStep 5: Evaluating classifier performance...")
    results = evaluate_classifier(classifier, X_test, y_test)
    
    # Check if any F-beta score exceeds 95%
    if any(score > 0.95 for score in results.values()):
        print("\nSuccess! Basic Hu moment features achieve at least one F-beta score exceeding 95%.")
    else:
        print("\nBasic features do not achieve the required 95% threshold. Trying enhanced features...")
        
        # Step 6: Extract enhanced features
        print("\nStep 6a: Extracting enhanced features from training images...")
        X_train_enhanced = extract_enhanced_features_from_paths(train_paths)
        
        print("\nStep 6b: Extracting enhanced features from testing images...")
        X_test_enhanced = extract_enhanced_features_from_paths(test_paths)
        
        # Step 7: Tune classifier with enhanced features
        print("\nStep 7: Tuning classifier with enhanced features...")
        tuned_classifier = tune_classifier(X_train_enhanced, y_train, classifier_type='svm')
        
        # Step 8: Evaluate tuned classifier
        print("\nStep 8: Evaluating tuned classifier with enhanced features...")
        enhanced_results = evaluate_classifier(tuned_classifier, X_test_enhanced, y_test)
        
        # Check if any F-beta score exceeds 95%
        if any(score > 0.95 for score in enhanced_results.values()):
            print("\nSuccess! Enhanced features achieve at least one F-beta score exceeding 95%.")
        else:
            print("\nWarning: Neither basic nor enhanced features achieve the 95% threshold.")
            print("Further optimization may be required.")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()