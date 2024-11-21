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

def main():
    image_path = "image/NakedTop01.jpg"  
    
    try:
        squares = visualize_detections(image_path, debug=True)
        print(f"Detected {len(squares)} squares in the image")
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()