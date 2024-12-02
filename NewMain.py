import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path):
    """
    Load and preprocess image with focus on clean square edges
    """
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
    else:
        image = image_path
        
    if image is None:
        raise ValueError(f"Could not load image")
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
    
    # Enhance contrast
    gray_image = cv2.equalizeHist(gray_image)
    
    # Apply bilateral filter to reduce noise while preserving edges
    denoised = cv2.bilateralFilter(gray_image, 9, 75, 75)
    
    return image, denoised

def detect_squares(image, min_area=150, max_area=10000, debug=False):
    """
    Detect squares using binary thresholding and contour analysis
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    squares = []
    debug_images = []
    
    # Try different threshold values
    for thresh_val in [100, 127, 150, 175, 200]:  # Add more threshold values
        # Binary threshold
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        if debug:
            debug_images.append((f"Threshold {thresh_val}", binary.copy()))
        
        # Find contours for both normal and inverted image
        for img in [binary, cv2.bitwise_not(binary)]:
            contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                    
                    # Check if the shape is approximately square
                    if len(approx) == 4:
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = float(w)/h
                        
                        # Check if it's square-like
                        if 0.7 < aspect_ratio < 1.3:
                            # Calculate the solidity (area / convex hull area)
                            hull = cv2.convexHull(contour)
                            hull_area = cv2.contourArea(hull)
                            solidity = float(area) / hull_area if hull_area > 0 else 0
                            
                            if solidity > 0.6:
                                squares.append({
                                    'contour': contour,
                                    'bbox': (x, y, w, h),
                                    'area': area,
                                    'aspect_ratio': aspect_ratio,
                                    'solidity': solidity
                                })
    
    # Remove overlapping detections
    filtered_squares = remove_overlapping_squares(squares)
    
    if debug:
        return filtered_squares, debug_images
    return filtered_squares

def remove_overlapping_squares(squares, iou_threshold=0.3):
    """
    Remove overlapping square detections
    """
    if not squares:
        return []
    
    # Sort squares by area
    squares = sorted(squares, key=lambda x: x['area'], reverse=True)
    
    filtered_squares = []
    while squares:
        current = squares.pop(0)
        keep = True
        
        i = 0
        while i < len(squares):
            if calculate_iou(current['bbox'], squares[i]['bbox']) > iou_threshold:
                squares.pop(i)
            else:
                i += 1
                
        if keep:
            filtered_squares.append(current)
    
    return filtered_squares

def calculate_iou(box1, box2):
    """Calculate Intersection over Union"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def visualize_detections(image_path, debug=False):
    """
    Visualize the square detection results
    """
    # Load and process image
    original_image, preprocessed = load_and_preprocess_image(image_path)
    
    # Detect squares
    if debug:
        squares, debug_images = detect_squares(preprocessed, debug=True)
    else:
        squares = detect_squares(preprocessed)
    
    # Create result visualization
    result_image = original_image.copy()
    debug_info = []
    
    # Draw detections
    for i, square in enumerate(squares, 1):
        # Draw contour
        cv2.drawContours(result_image, [square['contour']], -1, (0, 255, 0), 2)
        
        # Add detection info
        debug_info.append(
            f"Square {i}: Area={square['area']:.0f}, "
            f"AR={square['aspect_ratio']:.2f}, "
            f"Solidity={square['solidity']:.2f}"
        )
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(222)
    plt.imshow(preprocessed, cmap='gray')
    plt.title('Preprocessed Image')
    plt.axis('off')
    
    plt.subplot(223)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Detected Squares (Total: {len(squares)})')
    plt.axis('off')
    
    plt.subplot(224)
    plt.text(0.1, 0.5, '\n'.join(debug_info), fontsize=10)
    plt.axis('off')
    plt.title('Detection Properties')
    
    plt.tight_layout()
    plt.show()
    
    if debug:
        # Show debug images
        num_debug = len(debug_images)
        cols = min(3, num_debug)
        rows = (num_debug + cols - 1) // cols
        plt.figure(figsize=(5*cols, 5*rows))
        
        for i, (title, img) in enumerate(debug_images, 1):
            plt.subplot(rows, cols, i)
            plt.imshow(img, cmap='gray')
            plt.title(title)
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()
    
    return squares

def calculate_detection_accuracy(detected_squares, ground_truth_squares, iou_threshold=0.5):
    """
    Calculate detection accuracy metrics comparing detected squares against ground truth
    
    Args:
        detected_squares: List of detected square dictionaries with 'bbox' key
        ground_truth_squares: List of ground truth (x,y,w,h) tuples
        iou_threshold: Minimum IoU to consider a detection as correct
    
    Returns:
        Dictionary containing precision, recall, and F1 score
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Track which ground truth squares have been matched
    matched_gt = set()
    
    # For each detection, find best matching ground truth
    for det in detected_squares:
        best_iou = 0
        best_gt_idx = None
        
        for i, gt in enumerate(ground_truth_squares):
            if i in matched_gt:
                continue
                
            iou = calculate_iou(det['bbox'], gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        # Check if detection matches any ground truth
        if best_iou >= iou_threshold:
            true_positives += 1
            matched_gt.add(best_gt_idx)
        else:
            false_positives += 1
    
    # Count unmatched ground truth as false negatives
    false_negatives = len(ground_truth_squares) - len(matched_gt)
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

def visualize_detection_accuracy(image_path, ground_truth_squares, debug=False):
    """
    Visualize detection results with accuracy metrics
    """
    # Load and process image
    original_image, preprocessed = load_and_preprocess_image(image_path)
    
    # Detect squares
    detected_squares = detect_squares(preprocessed)
    
    # Calculate accuracy
    accuracy_metrics = calculate_detection_accuracy(detected_squares, ground_truth_squares)
    
    # Create visualization
    result_image = original_image.copy()
    
    # Draw detected squares in green
    for square in detected_squares:
        cv2.drawContours(result_image, [square['contour']], -1, (0, 255, 0), 2)
    
    # Draw ground truth squares in blue
    for gt in ground_truth_squares:
        x, y, w, h = gt
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(222)
    plt.imshow(preprocessed, cmap='gray')
    plt.title('Preprocessed Image')
    plt.axis('off')
    
    plt.subplot(223)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title('Detection Results\nGreen: Detected, Blue: Ground Truth')
    plt.axis('off')
    
    # Display metrics
    metrics_text = [
        f"Precision: {accuracy_metrics['precision']:.3f}",
        f"Recall: {accuracy_metrics['recall']:.3f}",
        f"F1 Score: {accuracy_metrics['f1']:.3f}",
        f"True Positives: {accuracy_metrics['true_positives']}",
        f"False Positives: {accuracy_metrics['false_positives']}",
        f"False Negatives: {accuracy_metrics['false_negatives']}"
    ]
    
    plt.subplot(224)
    plt.text(0.1, 0.5, '\n'.join(metrics_text), fontsize=10)
    plt.axis('off')
    plt.title('Accuracy Metrics')
    
    plt.tight_layout()
    plt.show()
    
    return detected_squares, accuracy_metrics

def main():
    image_path = "image/NakedTop01.jpg"
    
    # Define ground truth squares as (x, y, width, height)
    # You'll need to manually annotate these or load from a dataset
    ground_truth_squares = [
        (396, 803, 88, 81),
        (509, 799, 87, 86),
        (606, 726, 91, 89),
        (715, 718, 94, 85),
        (499, 699, 90, 83),
        (285, 688, 92, 86),
        (390, 687, 84, 86),
        (740, 617, 84, 85),
        (626, 606, 89, 92),
        (300, 586, 84, 87),
        (515, 576, 92, 92),
        (401, 575, 80, 88),
        (650, 509, 84, 85),
        (749, 496, 93, 88),
        (416, 478, 94, 91),
        (531, 473, 100, 93),
        (320, 467, 80, 84),
        (716, 384, 92, 89),
        (606, 367, 90, 87),
        (505, 365, 91, 93),
        (397, 358, 89, 88),
        (288, 357, 88, 88),
        (738, 286, 86, 88),
        (618, 262, 84, 83),
        (404, 256, 95, 94),
        (513, 255, 80, 81)
    ]
    
    
    try:
        detected_squares, accuracy = visualize_detection_accuracy(image_path, ground_truth_squares)
        print("\nAccuracy Metrics:")
        for metric, value in accuracy.items():
            print(f"{metric}: {value:.3f}")
            
    except Exception as e:
        print(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()