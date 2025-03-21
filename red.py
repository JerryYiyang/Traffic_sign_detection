import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def analyze_rgb(image_path):
    # Process image using normalized RGB
    image = Image.open(image_path).convert('RGB')
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
        return "Stop Sign"
    # Street signs typically have significant white areas
    else
        return "Street Sign"


print(improved_sign_classifier('car/selected_training_data/00004_00011_00019_png.rf.125a72cc106e0843e13d28102a138bc5.jpg'))