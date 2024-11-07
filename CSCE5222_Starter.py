# Required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Preprocessing Step 1: Load and Convert to Grayscale
def load_and_preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale if not already in grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization to adjust lighting
    equalized_image = cv2.equalizeHist(gray_image)
    
    return equalized_image

# Preprocessing Step 2: Noise Reduction
def apply_noise_reduction(image):
    # Apply Gaussian filter to reduce noise
    denoised_image = cv2.GaussianBlur(image, (5, 5), 0)
    return denoised_image

# Feature Extraction Step 1: Edge Detection for Pattern Identification
def detect_edges(image):
    # Use Canny edge detector to find edges in the image
    edges = cv2.Canny(image, 50, 150)
    return edges

# Feature Extraction Step 2: Hough Transform for Line Detection
def detect_lines(image):
    # Apply Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    
    # Draw detected lines on a blank canvas
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return line_image, lines

# Classification Step: Check for Stripe or Cross Pattern
def classify_pattern(lines):
    if lines is None:
        return "Bad"  # No lines detected
    elif len(lines) >= 5:  # Simple threshold for "Good" components (modify as needed)
        return "Good"
    else:
        return "Bad"

# Processing and Classification of a Single Image
def process_and_classify_image(image_path):
    # Preprocessing
    preprocessed_image = load_and_preprocess_image(image_path)
    denoised_image = apply_noise_reduction(preprocessed_image)
    
    # Feature Extraction
    edges = detect_edges(denoised_image)
    line_image, lines = detect_lines(edges)
    
    # Classification
    classification = classify_pattern(lines)
    
    # Display Results
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.title("Preprocessed Image")
    plt.imshow(preprocessed_image, cmap='gray')
    
    plt.subplot(1, 3, 2)
    plt.title("Edges Detected")
    plt.imshow(edges, cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.title("Detected Lines and Classification")
    plt.imshow(line_image, cmap='gray')
    plt.suptitle(f"Classification: {classification}")
    plt.show()
    
    return classification

# Batch Processing and Classification Metrics
def process_and_classify_batch(image_paths, ground_truth_labels):
    predictions = []  # To store the model predictions

    for idx, image_path in enumerate(image_paths):
        # Run the processing and classification for each image
        classification = process_and_classify_image(image_path)
        
        # Convert classification to binary label
        predicted_label = 1 if classification == "Good" else 0
        predictions.append(predicted_label)
        
        print(f"Image {idx + 1} - Ground Truth: {ground_truth_labels[idx]}, Predicted: {predicted_label}")

    # Calculate and print classification metrics
    accuracy = accuracy_score(ground_truth_labels, predictions)
    precision = precision_score(ground_truth_labels, predictions)
    recall = recall_score(ground_truth_labels, predictions)
    f1 = f1_score(ground_truth_labels, predictions)
    
    print("\nClassification Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

# Example usage with specified image paths
image_paths = [
    r"C:\Users\dunke\Pictures\Screenshots\Screenshot 2024-11-06 190656.png"
]  # Add more paths as needed

# Sample ground truth labels: 1 for "Good," 0 for "Bad" (replace with actual labels)
ground_truth_labels = [1]  # Update this list to match the number of images

# Run batch processing and evaluation
process_and_classify_batch(image_paths, ground_truth_labels)
