import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_process_image(image_path):
    """Load and process image for red square detection"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image")
    return image

def detect_red_squares(image, min_area=100, max_area=10000):
    """Detect red squares and return their coordinates"""
    # Create a mask for red squares
    lower_red = np.array([0, 0, 150])  # BGR values for red detection
    upper_red = np.array([100, 100, 255])
    red_mask = cv2.inRange(image, lower_red, upper_red)
    
    # Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    squares = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            
            # Check if the shape is approximately square
            if len(approx) >= 4:  # Allow for slight imperfections
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w)/h
                
                # Check if it's square-like
                if 0.7 < aspect_ratio < 1.3:
                    squares.append({
                        'bbox': (x, y, w, h),
                        'contour': contour
                    })
    
    return squares

def visualize_detections(image_path):
    """Visualize and return the detected red squares"""
    # Load image
    image = load_and_process_image(image_path)
    
    # Detect squares
    squares = detect_red_squares(image)
    
    # Create visualization
    result_image = image.copy()
    
    # Draw detections and collect coordinates
    coordinates = []
    for i, square in enumerate(squares, 1):
        x, y, w, h = square['bbox']
        coordinates.append(f"Square {i}: ({x}, {y}, {w}, {h})")
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(result_image, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Detected Red Squares (Total: {len(squares)})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print coordinates
    print("\nDetected Square Coordinates (x, y, width, height):")
    for coord in coordinates:
        print(coord)
    
    return squares

def main():
    # TODO: add rest of dataset
    image_path = "image/NakedTop01.png"
    
    try:
        squares = visualize_detections(image_path)
    except Exception as e:
        print(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()