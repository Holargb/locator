import cv2
import os

input_folder = "images"
output_folder = "Atest_RGB"

os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

for img_name in os.listdir(input_folder):
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Read the image (including alpha channel if present)
        
        # If the image is single-channel (grayscale), convert it to 3 channels
        if len(img.shape) == 2:  # Grayscale images have shape (H, W)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # If the image is RGBA (4 channels), remove the alpha channel
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, img)  # Save the converted RGB image

print("All images have been converted to RGB format!")