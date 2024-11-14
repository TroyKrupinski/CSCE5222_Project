# Required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.model_selection import StratifiedKFold

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

# Feature Extraction Step 1: Square Detection
def detect_squares(image):
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    squares = []
    for cnt in contours:
        # Approximate the contour to reduce the number of points
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Check if the approximated contour has 4 points and is convex
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # Compute the aspect ratio of the approximated contour
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            area = cv2.contourArea(approx)
            
            # Set thresholds for aspect ratio and area to filter out non-square quadrilaterals
            if 0.8 <= aspect_ratio <= 1.2 and area > 1000:
                squares.append(approx)
    return squares

# Feature Extraction Step 2: Edge Detection for Pattern Identification
def detect_edges(image):
    # Use Canny edge detector to find edges in the image
    edges = cv2.Canny(image, 50, 150)
    return edges

# Feature Extraction Step 3: Hough Transform for Line Detection
def detect_lines(image):
    # Apply Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
    
    # Draw detected lines on a blank canvas
    line_image = np.zeros_like(image)
    line_features = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
            
            # Calculate angle and length of the line
            angle = np.arctan2((y2 - y1), (x2 - x1)) * 180 / np.pi
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            line_features.append([angle, length])
    return line_image, lines, line_features

# Classification Step: Machine Learning Classifier (SVM)
def classify_pattern_svm(features, model):
    if not features:
        return "Bad"  # No lines detected
    
    # Convert features to numpy array
    features = np.array(features)
    
    # Predict using the trained model
    prediction = model.predict(features[:, 0].reshape(-1, 1))  # Using angle as feature
    
    # Majority vote
    if np.mean(prediction) >= 0.5:
        return "Good"
    else:
        return "Bad"

# Processing and Classification of a Single Image
def process_and_classify_image(image_path, model):
    # Preprocessing
    preprocessed_image = load_and_preprocess_image(image_path)
    denoised_image = apply_noise_reduction(preprocessed_image)
    
    # Square Detection
    squares = detect_squares(denoised_image)
    
    # Create an image to display detected squares
    squares_image = cv2.cvtColor(denoised_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(squares_image, squares, -1, (0, 255, 0), 2)
    
    # Default classification
    classification = "Bad"
    line_image = np.zeros_like(denoised_image)
    
    if squares:
        for square in squares:
            # Create a mask for the square region
            mask = np.zeros_like(denoised_image)
            cv2.drawContours(mask, [square], -1, 255, -1)
            
            # Extract the ROI using the mask
            roi = cv2.bitwise_and(denoised_image, denoised_image, mask=mask)
            
            # Feature Extraction within the square
            edges = detect_edges(roi)
            temp_line_image, lines, line_features = detect_lines(edges)
            line_image = cv2.bitwise_or(line_image, temp_line_image)
            
            # Classification based on detected lines within the square
            classification = classify_pattern_svm(line_features, model)
            
            # If a "Good" pattern is found in any square, classify the image as "Good"
            if classification == "Good":
                break
    else:
        # No squares detected, proceed with global pattern detection if necessary
        edges = detect_edges(denoised_image)
        line_image, lines, line_features = detect_lines(edges)
        classification = classify_pattern_svm(line_features, model)
    
    # Display Results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.title("Preprocessed Image")
    plt.imshow(preprocessed_image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title("Denoised Image")
    plt.imshow(denoised_image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.title("Detected Squares")
    plt.imshow(cv2.cvtColor(squares_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.title(f"Classification: {classification}")
    plt.imshow(line_image, cmap='gray')
    plt.axis('off')
    
    plt.show()
    
    return classification

# Training the Classifier (SVM)
def train_svm_classifier(training_image_paths, training_labels):
    feature_list = []
    label_list = []
    
    for idx, image_path in enumerate(training_image_paths):
        # Preprocessing
        preprocessed_image = load_and_preprocess_image(image_path)
        denoised_image = apply_noise_reduction(preprocessed_image)
        
        # Square Detection
        squares = detect_squares(denoised_image)
        
        if squares:
            for square in squares:
                # Create a mask for the square region
                mask = np.zeros_like(denoised_image)
                cv2.drawContours(mask, [square], -1, 255, -1)
                
                # Extract the ROI using the mask
                roi = cv2.bitwise_and(denoised_image, denoised_image, mask=mask)
                
                # Feature Extraction within the square
                edges = detect_edges(roi)
                _, _, line_features = detect_lines(edges)
                
                if line_features:
                    feature_list.extend(line_features)
                    label_list.extend([training_labels[idx]] * len(line_features))
                else:
                    # No lines detected; add a default feature
                    feature_list.append([0, 0])  # Angle 0, length 0
                    label_list.append(training_labels[idx])
        else:
            # No squares detected, proceed with global pattern detection
            edges = detect_edges(denoised_image)
            _, _, line_features = detect_lines(edges)
            if line_features:
                feature_list.extend(line_features)
                label_list.extend([training_labels[idx]] * len(line_features))
            else:
                # No lines detected; add a default feature
                feature_list.append([0, 0])  # Angle 0, length 0
                label_list.append(training_labels[idx])
    
    # Convert to numpy arrays
    feature_array = np.array(feature_list)
    label_array = np.array(label_list)
    
    # Check unique classes in labels
    unique_classes = np.unique(label_array)
    print("Unique classes in training labels:", unique_classes)
    if len(unique_classes) < 2:
        raise ValueError(f"The number of classes has to be greater than one; got {len(unique_classes)} class.")
    
    # Use angle as the feature for SVM
    X_train = feature_array[:, 0].reshape(-1, 1)  # Angle
    y_train = label_array
    
    # Train the SVM classifier
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    
    return svm_model

# Batch Processing and Classification Metrics using Cross-Validation
def cross_validate_model(image_paths, ground_truth_labels, n_splits=2):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_predictions = []
    all_true_labels = []
    
    for train_index, test_index in skf.split(image_paths, ground_truth_labels):
        training_image_paths = [image_paths[i] for i in train_index]
        testing_image_paths = [image_paths[i] for i in test_index]
        training_labels = [ground_truth_labels[i] for i in train_index]
        testing_labels = [ground_truth_labels[i] for i in test_index]
        
        # Train the SVM classifier
        svm_model = train_svm_classifier(training_image_paths, training_labels)
        
        # Test the classifier
        predictions = []
        for idx, image_path in enumerate(testing_image_paths):
            print(f"Processing Image: {image_path}")
            classification = process_and_classify_image(image_path, svm_model)
            predicted_label = 1 if classification == "Good" else 0
            predictions.append(predicted_label)
            print(f"Image - Ground Truth: {testing_labels[idx]}, Predicted: {predicted_label}")
        
        all_predictions.extend(predictions)
        all_true_labels.extend(testing_labels)
    
    # Calculate and print classification metrics
    accuracy = accuracy_score(all_true_labels, all_predictions)
    precision = precision_score(all_true_labels, all_predictions, zero_division=0)
    recall = recall_score(all_true_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_true_labels, all_predictions, zero_division=0)
    
    print("\nCross-Validation Classification Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

# Example usage with specified image paths
if __name__ == "__main__":
    # Replace the placeholders with your actual image paths and labels
    image_paths = [
        r"sample_shapes.jpg",
        r"example2.png",
        r"good.png",
        r"bad.png"
        # Add more paths as needed
    ]
    
    # Corresponding ground truth labels: 1 for "Good," 0 for "Bad"
    ground_truth_labels = [1, 1, 0, 0]  # Update this list to match the number of images

    # Verify that both classes are represented in the labels
    unique_labels = set(ground_truth_labels)
    print("Unique labels in ground truth:", unique_labels)
    if len(unique_labels) < 2:
        raise ValueError("Ground truth labels must include at least one 'Good' and one 'Bad' example.")

    # Perform cross-validation
    cross_validate_model(image_paths, ground_truth_labels, n_splits=2)