import numpy as np
from skimage import io
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def analyze_rgb(image):
    # Process image using normalized RGB
    # image = Image.open(image_path).convert('RGB')
    image_array = np.array(image).astype(float)
    
    # Get color channels
    r, g, b = image_array[:,:,0], image_array[:,:,1], image_array[:,:,2]
    
    # Normalize RGB
    rgb_sum = r + g + b + 1e-10
    r_norm = r/rgb_sum
    g_norm = g/rgb_sum
    b_norm = b/rgb_sum
    
    # Red pixel mask (as before)
    red_mask = (r_norm > 0.4) & (r_norm > g_norm * 1.5) & (r_norm > b_norm * 1.5)
    
    # White pixel mask (high and roughly equal values in all channels)
    white_mask = (r > 200) & (g > 200) & (b > 200) & \
                 (abs(r-g) < 30) & (abs(r-b) < 30) & (abs(g-b) < 30)
    
    # Calculate percentages
    red_percentage = np.sum(red_mask) / red_mask.size
    white_percentage = np.sum(white_mask) / white_mask.size
    
    # Key metric: ratio of red to white
    red_to_white_ratio = red_percentage / (white_percentage + 0.001)  # Avoid div by zero
    
    # Stop signs are predominantly red with little white
    if red_to_white_ratio > 4.0 and red_percentage > 0.15:
        return 1 
        # return "Stop Sign"
    # Street signs typically have significant white areas
    else:
        return 0
        # return "Street Sign"


# print(improved_sign_classifier('car/selected_training_data/00004_00011_00019_png.rf.125a72cc106e0843e13d28102a138bc5.jpg'))

def load_corresponding_colors(original_image_dir, mask_dir):
    count = 0
    color_features = []
    
    # extract the mask images
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.jpg', '.png'))]
    
    for mask_path in mask_files:
        if count % 100 == 0:
            print(f"Loading colors from masks for {count}/{len(mask_files)} images")
        base_name = os.path.splitext(mask_path)[0]
        count += 1
        if base_name.endswith("_mask"):
            original_base = base_name.replace("_mask", "")
        else:
            original_base = base_name

        original_file = None
        for ext in ['.jpg', '.png']:
            potential_file = original_base + ext
            if os.path.exists(os.path.join(original_image_dir, potential_file)):
                original_file = potential_file
                break
        
        if original_file is None:
            print(f"Could not find original image for {mask_path}")
            continue
        
        mask_image = io.imread(os.path.join(mask_dir, mask_path))
        original_image = io.imread(os.path.join(original_image_dir, original_file))

        mask_binary = mask_image > 0  # convert grayscale mask to binary
        # apply mask
        masked_image = original_image * np.repeat(mask_binary[:, :, np.newaxis], 3, axis=2)
        #cv2.imshow("Segmented Traffic Sign", masked_image)
        cv2.imwrite("COLORCHECK.jpg", masked_image) 

        sign = analyze_rgb(masked_image)
        color_features.append(sign)

    return color_features
