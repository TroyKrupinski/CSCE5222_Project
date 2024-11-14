import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def load_and_preprocess_image(image_path):
    """
    Load and preprocess an image
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
    
    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(gray_image)
    
    # Apply bilateral filter for noise reduction while preserving edges
    denoised_image = cv2.bilateralFilter(equalized_image, 9, 75, 75)
    
    return image, denoised_image

def calculate_iou(pred_box, gt_box):
    """
    Calculate Intersection over Union between prediction and ground truth boxes
    """
    x1 = max(pred_box[0], gt_box[0])
    y1 = max(pred_box[1], gt_box[1])
    x2 = min(pred_box[0] + pred_box[2], gt_box[0] + gt_box[2])
    y2 = min(pred_box[1] + pred_box[3], gt_box[1] + gt_box[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    pred_area = pred_box[2] * pred_box[3]
    gt_area = gt_box[2] * gt_box[3]
    union = pred_area + gt_area - intersection
    
    return intersection / union if union > 0 else 0

def detect_components(image):
    """
    Detect individual components in the image using multi-scale detection
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    components = []
    scales = [1.0, 1.5, 0.5]  # Multiple scales for detection
    
    for scale in scales:
        # Resize image for multi-scale detection
        width = int(gray.shape[1] * scale)
        height = int(gray.shape[0] * scale)
        scaled = cv2.resize(gray, (width, height))
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Noise removal
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Connected component analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opening, connectivity=8)
        
        # Filter components
        for i in range(1, num_labels):  # Skip background
            x = int(stats[i, cv2.CC_STAT_LEFT] / scale)
            y = int(stats[i, cv2.CC_STAT_TOP] / scale)
            w = int(stats[i, cv2.CC_STAT_WIDTH] / scale)
            h = int(stats[i, cv2.CC_STAT_HEIGHT] / scale)
            area = int(stats[i, cv2.CC_STAT_AREA] / (scale * scale))
            
            # Filter based on area and aspect ratio
            if area > 500 and 0.2 < w/h < 5:
                # Create contour from component
                component_mask = (labels == i).astype(np.uint8) * 255
                component_mask = cv2.resize(component_mask, (gray.shape[1], gray.shape[0]))
                contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, 
                                             cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    components.append((contours[0], (x, y, w, h)))
    
    return components

def extract_component_features(image, component):
    """
    Extract features from a detected component
    """
    contour, (x, y, w, h) = component
    
    # Ensure coordinates are within image bounds
    x = max(0, x)
    y = max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    
    if w <= 0 or h <= 0:
        return None
    
    roi = image[y:y+h, x:x+w]
    if roi.size == 0:
        return None
    
    # Calculate shape features
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Convert ROI to grayscale if needed
    if len(roi.shape) == 3:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        roi_gray = roi
    
    # Calculate intensity features
    mean_intensity = np.mean(roi_gray)
    std_intensity = np.std(roi_gray)
    
    # Calculate texture features
    gx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    texture_energy = np.mean(gradient_magnitude)
    
    # Create feature vector
    features = np.array([
        circularity,
        mean_intensity,
        std_intensity,
        texture_energy,
        w/h,  # aspect ratio
        area,
    ])
    
    return features

def evaluate_detection(image_paths, labels, test_size=0.2):
    """
    Evaluate the detection and classification system using SVM without IoU thresholding.
    """
    component_features = []
    component_labels = []
    
    print("Processing images and extracting features...")
    for idx, image_path in enumerate(image_paths):
        try:
            print(f"Processing {image_path}...")
            image, preprocessed = load_and_preprocess_image(image_path)
            components = detect_components(preprocessed)
            
            for component in components:
                features = extract_component_features(image, component)
                if features is not None:
                    component_features.append(features)
                    component_labels.append(labels[idx])  # Use provided labels directly
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue
    
    if len(component_features) < 2:
        print("Not enough components detected for evaluation")
        return None
    
    # Convert to numpy arrays
    X = np.array(component_features)
    y = np.array(component_labels)
    
    # Check for class diversity
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print("Insufficient class diversity for training. Ensure both 'good' and 'bad' components are present.")
        return None
    
    print(f"\nTotal components detected: {len(X)}")
    print(f"Class distribution: {np.bincount(y)}")
    
    try:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train classifier
        classifier = SVC(kernel='rbf', probability=True)
        classifier.fit(X_train, y_train)
        
        # Make predictions
        y_pred = classifier.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        print("\nEvaluation Results:")
        print("-" * 50)
        for metric, value in metrics.items():
            print(f"{metric.upper():10s}: {value:.3f}")
        
        return metrics
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return None


def visualize_results(image_path, save_path=None):
    """
    Visualize detection results with debug information
    """
    try:
        image, preprocessed = load_and_preprocess_image(image_path)
        components = detect_components(preprocessed)
        
        # Create debug visualization
        result_image = image.copy()
        debug_info = []
        
        for i, (contour, (x, y, w, h)) in enumerate(components):
            # Draw bounding box
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.drawContours(result_image, [contour], -1, (255, 0, 0), 1)
            
            # Calculate and store component properties
            area = cv2.contourArea(contour)
            aspect_ratio = w/h
            debug_info.append(f"Component {i+1}: Area={area:.0f}, AR={aspect_ratio:.2f}")
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(221)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(222)
        plt.imshow(preprocessed, cmap='gray')
        plt.title('Preprocessed Image')
        plt.axis('off')
        
        plt.subplot(223)
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Detected Components (Total: {len(components)})')
        plt.axis('off')
        
        plt.subplot(224)
        plt.text(0.1, 0.5, '\n'.join(debug_info), fontsize=10)
        plt.axis('off')
        plt.title('Component Properties')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    except Exception as e:
        print(f"Error visualizing {image_path}: {str(e)}")

def main():
    # Define image paths
    image_paths = [f"image/NakedTop{i:02d}.jpg" for i in range(1, 21)]
    
    # Example labels for each image (1 for "good" and 0 for "bad")
    # This list should match the ground truth classification based on visual inspection or existing labeled data.
    labels = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]
    
    # Evaluate the system
    metrics = evaluate_detection(image_paths, labels)
    
    # Optionally, visualize results for a few images
    for i, image_path in enumerate(image_paths[:5]):
        visualize_results(image_path, f"results_image_{i+1}.png")

if __name__ == "__main__":
    main()
