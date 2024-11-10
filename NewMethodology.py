import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_sample_image():
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (150, 150), (255, 255, 255), -1)  # Square
    cv2.rectangle(image, (200, 50), (300, 130), (255, 255, 255), -1)  # Rectangle
    cv2.circle(image, (400, 100), 50, (255, 255, 255), -1)            # Circle
    cv2.rectangle(image, (100, 200), (180, 300), (255, 255, 255), -1) # Rectangle
    cv2.rectangle(image, (250, 250), (350, 350), (255, 255, 255), -1) # Square
    cv2.imwrite('sample_shapes.jpg', image)
    return 'sample_shapes.jpg'

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray_image

def find_contours(gray_image):
    edges = cv2.Canny(gray_image, 50, 150)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    return contours, contour_image

def approximate_polygons(contours, gray_image):
    approx_polygons = []
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        approx_polygons.append(approx)
    polygon_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(polygon_image, approx_polygons, -1, (0, 0, 255), 2)
    return approx_polygons, polygon_image

def filter_squares(approx_polygons, gray_image):
    squares = []
    for approx in approx_polygons:
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            area = cv2.contourArea(approx)
            if 0.8 <= aspect_ratio <= 1.2 and area > 1000:
                squares.append(approx)
    squares_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(squares_image, squares, -1, (255, 0, 0), 2)
    return squares, squares_image

def isolate_square_regions(squares, gray_image):
    isolated_images = []
    for i, square in enumerate(squares):
        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask, [square], -1, 255, -1)
        isolated = cv2.bitwise_and(gray_image, gray_image, mask=mask)
        isolated_images.append(isolated)
    return isolated_images

if __name__ == "__main__":
    # Create sample image
    image_path = create_sample_image()
    
    # Load and preprocess the image
    original_image, gray_image = load_and_preprocess_image(image_path)
    
    # Step 1: Find Contours
    contours, contour_image = find_contours(gray_image)
    
    # Step 2: Approximate Contours to Polygons
    approx_polygons, polygon_image = approximate_polygons(contours, gray_image)
    
    # Step 3: Filter for Quadrilaterals (Squares)
    squares, squares_image = filter_squares(approx_polygons, gray_image)
    
    # Step 4: Isolate Square Regions
    isolated_images = isolate_square_regions(squares, gray_image)
    
    # Display the images at each step
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 3, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.title('Grayscale Image')
    plt.imshow(gray_image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.title('Contours Found')
    plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.title('Approximated Polygons')
    plt.imshow(cv2.cvtColor(polygon_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.title('Filtered Squares')
    plt.imshow(cv2.cvtColor(squares_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.title('Isolated Square Regions')
    if isolated_images:
        # Show the first isolated square region
        plt.imshow(isolated_images[0], cmap='gray')
    else:
        plt.text(0.5, 0.5, 'No squares detected', horizontalalignment='center', verticalalignment='center')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
