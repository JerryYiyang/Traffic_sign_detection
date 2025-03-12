from skimage import io, color, filters, util
from skimage.feature import hog
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import numpy as np
import sys

def plot_clusters_2d(feature_vectors, labels):
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(feature_vectors)

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(label="Cluster")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("K-Means Clustering (2D PCA)")
    plt.show()

def kmeans(feature_vectors):

    feature_vectors = np.array(feature_vectors)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    labels = kmeans.fit_predict(feature_vectors)
    plot_clusters_2d(feature_vectors, labels)
    return kmeans, labels

# do we want to use different features for the clustering? or just the key points?
def hog_feature_extraction(output_dir):
    count = 0
    feature_vectors = []
    processed_imgs = os.listdir(output_dir)
    total_imgs = min(300, len(processed_imgs))  # Process max 300 images
    
    for img_file in processed_imgs:
        if count >= 300:
            break
            
        if count % 100 == 0:
            print(f"At {count}/{total_imgs} images")
            
        try:
            # Read in preprocessed images
            img_path = os.path.join(output_dir, img_file)
            image = io.imread(img_path)
            
            # Extract HOG features but don't visualize unless needed
            features = hog(image, orientations=9, pixels_per_cell=(8, 8), 
                          cells_per_block=(2, 2), visualize=False)
            
            feature_vectors.append(features)
            count += 1
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            
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
    # if len(sys.argv) != 3:
    #     print("Usage: python preprocess.py image_dir output_dir")
    #     exit(1)
    
    image_dir = 'car/train/images'
    output_dir = 'output_dir'

    preprocess(image_dir, output_dir)
    feature_vectors = hog_feature_extraction(output_dir)
    kmeans(feature_vectors)