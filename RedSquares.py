import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_and_process_image(image_path):
    """Load and process image for red square detection"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image")
    return image

def detect_red_squares(image, min_area=100, max_area=10000):
    """Detect red squares and return their coordinates"""
    lower_red = np.array([0, 0, 150])
    upper_red = np.array([100, 100, 255])
    red_mask = cv2.inRange(image, lower_red, upper_red)
    
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    squares = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w)/h
                
                if 0.7 < aspect_ratio < 1.3:
                    squares.append({
                        'bbox': (x, y, w, h),
                        'contour': contour
                    })
    
    return squares

def visualize_detections(image_path, save_visualization=False):
    """Visualize and return the detected red squares"""
    image = load_and_process_image(image_path)
    squares = detect_red_squares(image)
    result_image = image.copy()
    
    coordinates = []
    for i, square in enumerate(squares, 1):
        x, y, w, h = square['bbox']
        coordinates.append((x, y, w, h))
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(result_image, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Detected Red Squares (Total: {len(squares)})')
    plt.axis('off')
    
    if save_visualization:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        plt.savefig(f'visualizations/{base_name}_detection.png')
    
    plt.show()
    
    return coordinates

def process_all_images():
    """Process all images and generate ground truth data"""
    # Use the correct base path and file extension
    base_path = r"C:\Users\dunke\Desktop\CSCE5222_Project\Image\ground"
    
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    all_ground_truths = {}
    
    # Process each image
    for i in range(1, 21):
        image_name = f"NakedTop{i:02d}.png"  # Changed to .png
        image_path = os.path.join(base_path, image_name)
        
        print(f"\nProcessing image: {image_name}")
        
        try:
            coordinates = visualize_detections(image_path, save_visualization=True)
            all_ground_truths[image_name] = coordinates
            
            # Print coordinates in a format ready for ground truth
            print(f"\nGround truth for {image_name}:")
            print("ground_truth_squares = [")
            for x, y, w, h in coordinates:
                print(f"    ({x}, {y}, {w}, {h}),")
            print("]\n")
            
        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")
    
    # Generate complete ground truth file
    generate_ground_truth_file(all_ground_truths)

def generate_ground_truth_file(all_ground_truths):
    """Generate a Python file containing all ground truths"""
    with open('ground_truths.py', 'w') as f:
        f.write("# Ground truth coordinates for all images\n\n")
        f.write("ground_truths = {\n")
        
        for image_name, coordinates in all_ground_truths.items():
            f.write(f"    '{image_name}': [\n")
            for x, y, w, h in coordinates:
                f.write(f"        ({x}, {y}, {w}, {h}),\n")
            f.write("    ],\n")
        
        f.write("}\n")

def main():
    process_all_images()

if __name__ == "__main__":
    main()