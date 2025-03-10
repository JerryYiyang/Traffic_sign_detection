from skimage.feature import hog
from skimage import io, color, exposure, feature, filters, transform, util
import os
import numpy as np
import matplotlib.pyplot as plt

def preprocess(image, size=(128,128)):
    results = {}

    resized = transform.resize(image, size, anti_aliasing=True)
    gray = color.rgb2gray(resized)

    results['gray'] = util.img_as_ubyte(gray)
    enhanced = exposure.equalize_adapthist(gray, clip_limit=0.03)
    results['enhanced'] = util.img_as_ubyte(enhanced)

    smoothed = filters.gaussian(enhanced, sigma=1, preserve_range=True)
    results['smoothed'] = util.img_as_ubyte(smoothed)
    
    return results

def batch_preprocess(image_dir, label_dir, output_dir, size=(128, 128)):

    processed_data = {'images': [], 'labels': [],
                      'filename': []}
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    image_files = os.listdir(image_dir)
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, base_name + '.txt')

        if not os.path.exists(label_path):
            continue
        with open(label_path, 'r') as f:
            label_data = f.readlines()
        
        image = io.imread(img_path)
        processed = preprocess(image)
        