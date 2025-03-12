from skimage import io, color, filters, util
from skimage.feature import hog
import matplotlib.pyplot as plt
import os
import sys

# def kmeans(feature_vectors):
#     return

# do we want to use different features for the clustering? or just the key points?
def hog_feature_extraction(output_dir):
    feature_vectors = []
    processed_imgs = os.listdir(output_dir)
    for i in range(len(processed_imgs)):
        try: 
            # Read in preprocessed images
            image = io.imread(processed_imgs)
            # Extract HOG features
            features, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                                    cells_per_block=(2, 2), visualize=True)
            feature_vectors.append(features)
        except Exception as e:
            print(f"Error processing {image}: {str(e)}")
    return feature_vectors

def preprocess(image_dir, output_dir):
    print("starting")
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    image_files = os.listdir(image_dir)
    count = 0
    num_images = len(image_files)
    for img_file in image_files:
        if count > 300:
            break
        if count % 100 == 0:
            print(f"At {count}/{num_images} images")
        count += 1
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(image_dir, img_file)
        try:
            image = io.imread(img_path)
            gray = color.rgb2gray(image)
            smoothed = filters.gaussian(gray, sigma=1, preserve_range=True)
            processed = util.img_as_ubyte(smoothed)
            if output_dir is not None:
                output_path = os.path.join(output_dir, f"{base_name}.jpg")
                io.imsave(output_path, processed, check_contrast=False)
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess.py image_dir output_dir")
        exit(1)
    
    image_dir = sys.argv[1]
    output_dir = sys.argv[2]

    preprocess(image_dir, output_dir)
    feature_vectors = hog_feature_extraction(output_dir)
    # kmeans(feature_vectors)