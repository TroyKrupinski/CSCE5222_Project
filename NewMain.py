import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
def load_and_preprocess_image(image_path):
    """
    Enhanced preprocessing to improve square detection
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
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray_image)
    
    # Apply bilateral filter to reduce noise while preserving edges
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    return image, denoised

def detect_squares(image, min_area=2000, max_area=25000, debug=False):
    """
    Square detection with focus on improved recall
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    squares = []
    debug_images = []
    
    # More threshold values for better detection
    threshold_values = np.linspace(50, 200, 10).astype(np.int32)
    
    for thresh_val in threshold_values:
        # Try both normal and inverted thresholds
        for thresh_type in [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV]:
            _, binary = cv2.threshold(gray, thresh_val, 255, thresh_type)
            
            # Light morphological operations
            kernel = np.ones((2,2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.08 * peri, True)  # More lenient approximation
                    
                    if len(approx) in [4, 5]:  # Accept slightly imperfect squares
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = float(w)/h
                        
                        # More lenient aspect ratio
                        if 0.6 < aspect_ratio < 1.4:
                            hull = cv2.convexHull(contour)
                            hull_area = cv2.contourArea(hull)
                            solidity = float(area) / hull_area if hull_area > 0 else 0
                            
                            # More lenient solidity threshold
                            if solidity > 0.5:
                                squares.append({
                                    'contour': contour,
                                    'bbox': (x, y, w, h),
                                    'area': area,
                                    'aspect_ratio': aspect_ratio,
                                    'solidity': solidity
                                })
    
    # Remove overlapping detections
    filtered_squares = remove_overlapping_squares(squares, iou_threshold=0.4)
    
    if debug:
        return filtered_squares, debug_images
    return filtered_squares

def remove_overlapping_squares(squares, iou_threshold=0.4):
    """
    Remove overlapping squares with more lenient threshold
    """
    if not squares:
        return []
    
    # Sort squares by quality (solidity and aspect ratio)
    squares = sorted(squares, 
                    key=lambda x: (x['solidity'] * (1 - abs(1 - x['aspect_ratio']))),
                    reverse=True)
    
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
        f"True Negatives: {len(ground_truth_squares) - accuracy_metrics['false_negatives']}",
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


    
    # Define ground truth squares as (x, y, width, height)
    # You'll need to manually annotate these or load from a dataset

def main():
    # Use relative path for local images
    base_path = "Image"
    ground_truth_squares = {
        "NakedTop01.png": [
            (396, 803, 91, 83), (505, 793, 95, 88), (608, 726, 94, 86),
            (716, 714, 94, 89), (494, 699, 95, 81), (289, 690, 89, 86),
            (391, 688, 89, 91), (743, 622, 78, 85), (626, 612, 88, 77),
            (300, 587, 85, 88), (517, 580, 87, 97), (402, 577, 88, 93),
            (860, 508, 83, 83), (647, 507, 86, 91), (745, 496, 85, 91),
            (539, 475, 83, 88), (420, 475, 82, 88), (314, 468, 87, 89),
            (717, 390, 86, 83), (506, 372, 89, 87), (609, 370, 88, 80),
            (395, 360, 86, 84), (290, 360, 83, 81), (738, 288, 88, 87),
            (509, 262, 85, 86), (407, 260, 86, 79), (614, 258, 88, 86)
        ],
        "NakedTop02.png": [
            (503, 794, 89, 94), (392, 791, 99, 90), (606, 725, 90, 87),
            (719, 721, 85, 84), (500, 697, 84, 85), (384, 690, 95, 80),
            (289, 686, 81, 82), (734, 612, 83, 91), (630, 610, 85, 85),
            (302, 591, 81, 82), (511, 578, 90, 87), (861, 510, 92, 86),
            (649, 508, 78, 79), (743, 498, 92, 95), (533, 478, 98, 90),
            (320, 466, 84, 89), (720, 390, 92, 85), (510, 374, 91, 91),
            (611, 371, 87, 83), (288, 366, 91, 92), (402, 359, 90, 92),
            (738, 293, 85, 80), (405, 262, 89, 85), (620, 257, 84, 89),
            (509, 252, 91, 95)
        ],
        "NakedTop03.png": [
            (509, 797, 91, 91), (395, 790, 91, 92), (609, 726, 94, 90),
            (714, 710, 96, 98), (495, 694, 89, 87), (285, 694, 93, 81),
            (393, 690, 82, 78), (738, 615, 90, 88), (624, 604, 100, 91),
            (300, 591, 85, 93), (515, 580, 95, 94), (397, 579, 87, 84),
            (861, 507, 86, 90), (648, 502, 91, 97), (750, 499, 85, 85),
            (422, 481, 88, 89), (536, 478, 86, 86), (313, 469, 89, 91),
            (720, 387, 90, 86), (506, 372, 86, 77), (600, 369, 102, 90),
            (281, 361, 96, 83), (396, 355, 90, 94), (737, 289, 85, 83),
            (615, 261, 86, 84), (504, 261, 89, 82), (405, 259, 92, 85)
        ],
        "NakedTop04.png": [
            (397, 799, 84, 75), (506, 796, 95, 83), (609, 722, 88, 90),
            (719, 717, 84, 85), (499, 694, 87, 84), (287, 687, 87, 90),
            (391, 683, 83, 87), (740, 614, 83, 90), (630, 611, 82, 79),
            (301, 586, 83, 83), (397, 580, 84, 81), (519, 575, 85, 85),
            (861, 508, 94, 89), (649, 508, 87, 88), (751, 501, 87, 84),
            (542, 478, 86, 88), (420, 476, 91, 91), (319, 464, 90, 96),
            (715, 388, 88, 88), (506, 369, 86, 85), (399, 364, 80, 76),
            (607, 363, 82, 91), (284, 359, 88, 82), (740, 286, 83, 95),
            (618, 262, 85, 90), (409, 259, 89, 92), (509, 257, 90, 94)
        ],
        "NakedTop05.png": [
            (512, 799, 86, 83), (404, 798, 83, 84), (606, 729, 91, 88),
            (719, 719, 92, 86), (500, 693, 86, 91), (391, 689, 78, 83),
            (289, 687, 90, 85), (738, 612, 89, 94), (630, 603, 88, 91),
            (398, 581, 94, 89), (517, 575, 86, 93), (299, 575, 90, 95),
            (645, 508, 94, 85), (860, 496, 80, 96), (752, 495, 75, 85),
            (540, 477, 85, 85), (415, 477, 89, 85), (319, 470, 91, 85),
            (716, 392, 96, 84), (612, 369, 84, 84), (506, 364, 87, 89),
            (398, 358, 89, 87), (285, 356, 97, 87), (739, 286, 90, 89),
            (510, 263, 88, 92), (405, 257, 97, 93), (621, 255, 82, 92)
        ],
        "NakedTop06.png": [
            (512, 800, 84, 81), (398, 796, 93, 91), (607, 728, 95, 87),
            (719, 715, 92, 85), (497, 693, 92, 93), (386, 691, 88, 90),
            (284, 687, 93, 93), (740, 618, 90, 89), (627, 607, 89, 84),
            (296, 588, 89, 85), (398, 579, 89, 86), (518, 576, 83, 87),
            (859, 508, 84, 81), (649, 507, 88, 87), (750, 501, 86, 84),
            (539, 481, 95, 85), (422, 477, 87, 83), (314, 466, 102, 94),
            (719, 393, 83, 84), (612, 374, 79, 81), (506, 367, 100, 96),
            (397, 362, 79, 79), (289, 362, 84, 83), (738, 288, 80, 87),
            (512, 258, 78, 84), (411, 258, 87, 91), (617, 255, 83, 89)
        ],
        "NakedTop07.png": [
            (400, 803, 84, 80), (510, 800, 92, 84), (610, 725, 92, 87),
            (719, 719, 90, 81), (502, 699, 92, 83), (288, 688, 94, 81),
            (390, 687, 92, 82), (630, 610, 81, 83), (729, 604, 97, 95),
            (296, 587, 89, 91), (398, 586, 94, 74), (513, 575, 98, 89),
            (859, 503, 95, 88), (646, 501, 92, 96), (744, 494, 86, 89),
            (536, 481, 91, 86), (721, 386, 90, 89), (608, 371, 85, 81),
            (403, 369, 89, 77), (511, 368, 83, 85), (284, 361, 101, 82),
            (739, 295, 83, 76), (506, 263, 90, 82), (617, 259, 87, 82),
            (403, 256, 97, 95)
        ],
        "NakedTop08.png": [
            (503, 799, 90, 84), (393, 794, 93, 98), (609, 722, 86, 89),
            (719, 720, 84, 83), (501, 699, 83, 81), (291, 692, 84, 86),
            (392, 687, 84, 86), (740, 618, 83, 83), (631, 608, 86, 88),
            (295, 590, 88, 86), (520, 576, 97, 91), (402, 576, 89, 85),
            (652, 511, 77, 79), (864, 509, 80, 81), (750, 500, 81, 78),
            (415, 482, 88, 86), (538, 480, 89, 85), (320, 470, 84, 89),
            (721, 392, 85, 83), (610, 369, 85, 83), (510, 368, 93, 95),
            (400, 361, 89, 84), (288, 360, 99, 86), (733, 293, 93, 85),
            (401, 257, 95, 87), (615, 255, 90, 91), (509, 254, 89, 95)
        ],
        "NakedTop09.png": [
            (510, 797, 89, 85), (388, 791, 105, 92), (604, 727, 94, 87),
            (719, 721, 92, 79), (504, 699, 84, 81), (286, 690, 93, 82),
            (390, 687, 90, 81), (738, 615, 91, 93), (627, 608, 86, 84),
            (292, 584, 98, 95), (518, 576, 92, 86), (402, 575, 82, 90),
            (854, 509, 92, 87), (648, 506, 88, 89), (751, 498, 85, 89),
            (422, 482, 88, 84), (535, 476, 97, 90), (316, 467, 88, 92),
            (720, 392, 92, 81), (601, 373, 99, 84), (506, 372, 90, 90),
            (398, 362, 93, 83), (289, 362, 94, 81), (735, 293, 95, 86),
            (508, 263, 92, 85), (616, 257, 91, 89), (410, 257, 97, 92)
        ],
        "NakedTop10.png": [
            (509, 797, 85, 88), (397, 792, 90, 87), (609, 729, 89, 80),
            (726, 718, 84, 82), (500, 698, 98, 84), (284, 692, 87, 74),
            (393, 684, 88, 88), (738, 618, 85, 82), (626, 605, 84, 89),
            (299, 586, 89, 91), (516, 575, 84, 89), (397, 575, 87, 90),
            (857, 508, 90, 85), (643, 507, 90, 86), (746, 497, 89, 88),
            (422, 479, 86, 88), (536, 478, 93, 89), (314, 467, 100, 87),
            (716, 386, 99, 88), (506, 370, 93, 90), (605, 367, 92, 90),
            (399, 361, 87, 83), (283, 360, 95, 91), (734, 289, 96, 89),
            (510, 266, 90, 87), (620, 259, 84, 87), (410, 257, 95, 91)
        ],
        "NakedTop11.png": [
            (511, 801, 88, 85), (395, 796, 98, 94), (607, 726, 90, 88),
            (716, 718, 95, 85), (498, 702, 86, 78), (391, 684, 93, 90),
            (288, 684, 84, 88), (738, 619, 89, 83), (626, 612, 89, 86),
            (297, 588, 91, 87), (517, 579, 90, 87), (403, 577, 95, 93),
            (650, 510, 93, 83), (858, 508, 91, 83), (752, 499, 83, 82),
            (537, 480, 94, 84), (420, 474, 88, 94), (315, 467, 93, 91),
            (719, 388, 89, 88), (401, 360, 87, 84), (291, 357, 91, 86),
            (735, 291, 91, 84), (611, 258, 93, 89), (407, 258, 95, 88),
            (509, 252, 83, 90)
        ],
        "NakedTop12.png": [
            (506, 798, 88, 86), (396, 797, 90, 90), (610, 720, 86, 99),
            (718, 709, 95, 98), (495, 697, 89, 86), (282, 691, 97, 91),
            (384, 685, 93, 88), (742, 618, 83, 86), (629, 612, 83, 82),
            (300, 591, 89, 84), (509, 580, 102, 82), (398, 576, 83, 88),
            (861, 508, 94, 90), (653, 508, 85, 88), (754, 502, 85, 91),
            (537, 481, 95, 89), (413, 479, 92, 84), (312, 461, 97, 96),
            (715, 388, 94, 88), (608, 370, 94, 82), (506, 370, 95, 83),
            (288, 359, 94, 79), (398, 357, 87, 86), (738, 288, 92, 87),
            (617, 259, 92, 86), (412, 258, 87, 89), (513, 256, 77, 89)
        ],
        "NakedTop13.png": [
            (507, 802, 95, 82), (396, 798, 87, 82), (604, 730, 97, 80),
            (715, 713, 95, 90), (501, 700, 91, 85), (390, 690, 91, 87),
            (284, 685, 91, 91), (733, 613, 97, 94), (623, 610, 92, 88),
            (290, 586, 98, 86), (517, 580, 87, 82), (393, 580, 96, 85),
            (647, 505, 87, 87), (856, 501, 87, 92), (750, 498, 87, 80),
            (536, 480, 95, 83), (425, 472, 84, 90), (320, 467, 92, 88),
            (720, 391, 93, 83), (615, 372, 79, 81), (505, 367, 88, 91),
            (396, 359, 94, 90), (284, 355, 91, 88), (734, 283, 92, 94),
            (616, 263, 90, 77), (503, 258, 92, 90), (400, 258, 92, 86)
        ],
        "NakedTop14.png": [
            (508, 796, 90, 90), (391, 791, 105, 93), (606, 727, 95, 88),
            (713, 719, 95, 84), (497, 699, 86, 87), (282, 689, 94, 84),
            (385, 683, 91, 94), (742, 619, 80, 87), (630, 610, 88, 80),
            (298, 588, 94, 90), (402, 579, 89, 85), (512, 577, 95, 87),
            (649, 514, 83, 84), (861, 504, 88, 90), (748, 498, 81, 83),
            (422, 481, 78, 82), (538, 478, 92, 89), (308, 469, 103, 86),
            (718, 387, 89, 89), (610, 367, 85, 83), (508, 366, 88, 88),
            (397, 361, 92, 82), (284, 354, 104, 94), (736, 288, 88, 83),
            (509, 258, 83, 83), (406, 258, 89, 87), (610, 253, 97, 93)
        ],
        "NakedTop15.png": [
            (503, 796, 98, 90), (397, 795, 90, 90), (608, 726, 92, 92),
            (717, 714, 89, 89), (496, 702, 93, 83), (390, 690, 83, 88),
            (280, 690, 104, 89), (739, 623, 88, 75), (629, 612, 87, 86),
            (299, 587, 92, 90), (517, 581, 88, 85), (396, 576, 96, 88),
            (646, 510, 90, 86), (860, 505, 90, 93), (753, 497, 90, 90),
            (535, 477, 88, 90), (419, 475, 84, 86), (314, 465, 93, 92),
            (721, 388, 87, 91), (609, 370, 91, 79), (509, 366, 88, 90),
            (286, 361, 95, 83), (395, 354, 90, 89), (742, 292, 79, 84),
            (619, 264, 83, 80), (510, 261, 91, 80), (406, 257, 92, 88)
        ],
        "NakedTop16.png": [
            (509, 798, 90, 86), (397, 795, 92, 86), (610, 725, 82, 88),
            (720, 714, 84, 90), (495, 695, 92, 88), (281, 690, 101, 84),
            (390, 689, 87, 85), (739, 623, 89, 81), (630, 610, 92, 86),
            (292, 588, 92, 90), (398, 579, 90, 80), (519, 575, 85, 93),
            (647, 510, 88, 88), (862, 504, 88, 90), (748, 497, 90, 85),
            (535, 480, 89, 84), (418, 474, 93, 89), (315, 466, 91, 86),
            (715, 388, 86, 85), (609, 367, 91, 87), (509, 366, 92, 89),
            (394, 357, 93, 85), (291, 357, 87, 91), (736, 288, 93, 90),
            (511, 263, 86, 88), (405, 261, 87, 80), (616, 258, 91, 92)
        ],
        "NakedTop17.png": [
            (507, 800, 94, 83), (396, 797, 94, 88), (609, 730, 94, 84),
            (722, 718, 84, 90), (495, 700, 93, 80), (391, 695, 90, 79),
            (283, 692, 93, 82), (740, 613, 89, 89), (631, 607, 85, 89),
            (295, 586, 88, 90), (402, 577, 84, 87), (522, 576, 79, 91),
            (861, 515, 91, 79), (650, 505, 92, 88), (751, 500, 94, 82),
            (422, 481, 89, 80), (535, 479, 88, 91), (317, 469, 93, 90),
            (721, 389, 84, 83), (608, 370, 90, 81), (506, 367, 94, 90),
            (284, 360, 93, 85), (392, 356, 98, 97), (735, 290, 92, 86),
            (624, 259, 86, 88), (506, 259, 94, 90), (404, 256, 87, 89)
        ],
        "NakedTop18.png": [
            (396, 797, 92, 89), (511, 796, 89, 86), (608, 727, 89, 88),
            (716, 712, 94, 93), (499, 701, 89, 84), (391, 690, 84, 84),
            (287, 689, 95, 85), (735, 617, 97, 85), (625, 607, 96, 88),
            (290, 584, 100, 93), (398, 582, 89, 76), (516, 577, 88, 83),
            (858, 506, 85, 89), (650, 505, 84, 87), (744, 496, 94, 87),
            (542, 478, 83, 86), (416, 475, 95, 92), (314, 467, 89, 90),
            (710, 388, 93, 84), (504, 367, 91, 87), (607, 366, 85, 90),
            (283, 360, 88, 87), (392, 358, 92, 89), (733, 288, 90, 88),
            (612, 257, 92, 89), (512, 256, 80, 87), (399, 252, 97, 95)
        ],
        "NakedTop19.png": [
            (397, 799, 93, 88), (509, 798, 89, 83), (608, 727, 89, 88),
            (713, 715, 99, 87), (498, 698, 86, 83), (283, 685, 95, 89),
            (392, 682, 83, 92), (740, 614, 87, 87), (626, 600, 89, 90),
            (293, 582, 90, 96), (516, 578, 86, 83), (398, 575, 86, 90),
            (644, 509, 89, 86), (857, 506, 86, 87), (749, 497, 86, 86),
            (415, 480, 97, 81), (541, 478, 85, 84), (317, 473, 87, 84),
            (715, 388, 96, 89), (609, 371, 79, 85), (506, 367, 94, 89),
            (396, 365, 89, 85), (290, 355, 86, 92), (735, 289, 92, 88),
            (619, 260, 90, 85), (507, 256, 88, 87), (407, 256, 90, 92)
        ],
        "NakedTop20.png": [
            (506, 801, 95, 83), (396, 800, 94, 89), (608, 727, 87, 87),
            (716, 716, 89, 87), (495, 698, 93, 88), (385, 689, 90, 87),
            (284, 689, 90, 87), (740, 620, 88, 86), (624, 607, 92, 90),
            (294, 587, 89, 91), (398, 577, 92, 91), (512, 573, 92, 92),
            (644, 507, 93, 88), (859, 506, 87, 90), (750, 501, 85, 85),
            (416, 476, 94, 88), (531, 473, 98, 92), (313, 470, 89, 89),
            (716, 389, 94, 89), (609, 367, 88, 89), (506, 364, 92, 92),
            (398, 358, 89, 91), (288, 355, 95, 95), (739, 286, 89, 86),
            (506, 257, 93, 88), (615, 256, 94, 87), (407, 253, 92, 95)
        ]
    }
    
    
    # Track overall metrics across all images
    overall_metrics = {
        'precision': 0,
        'recall': 0,
        'f1': 0,
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0
    }
    
    # Process each image
    successful_images = 0
    
    # Process all images from 1 to 20
    for i in range(1, 21):
        image_name = f"NakedTop{i:02d}.jpg"  # Local images are .jpg
        ground_truth_name = f"NakedTop{i:02d}.png"  # Ground truth keys are .png
        image_path = os.path.join(base_path, image_name)
        
        print(f"\nProcessing {image_name}")
        print(f"Image path: {image_path}")
        
        try:
            # Get ground truth for this image
            if ground_truth_name not in ground_truth_squares:
                print(f"No ground truth found for {ground_truth_name}")
                continue
                
            ground_truth = ground_truth_squares[ground_truth_name]
            
            # Run detection and accuracy comparison
            detected_squares, accuracy = visualize_detection_accuracy(image_path, ground_truth)
            
            # Print individual image metrics
            print(f"\nAccuracy Metrics for {image_name}:")
            for metric, value in accuracy.items():
                print(f"{metric}: {value:.3f}")
            
            # Print detection details
            print(f"\nDetection Details:")
            print(f"Found {len(detected_squares)} squares")
            print(f"Ground truth has {len(ground_truth)} squares")
            
            # Add to overall metrics
            for metric in overall_metrics:
                if isinstance(accuracy[metric], (int, float)):
                    overall_metrics[metric] += accuracy[metric]
            
            successful_images += 1
            
        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")
    
    # Calculate and display average metrics
    if successful_images > 0:
        print("\n=== Overall Average Metrics ===")
        for metric in overall_metrics:
            avg_value = overall_metrics[metric] / successful_images
            print(f"Average {metric}: {avg_value:.3f}")
        
        print(f"\nSuccessfully processed {successful_images} out of 20 images")

if __name__ == "__main__":
    main()