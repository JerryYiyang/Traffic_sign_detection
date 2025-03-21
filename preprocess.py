from skimage import io, color, filters, util
from skimage.feature import hog
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import numpy as np
import sys
import cv2
import easyocr
import red

def segment_images(dir):
    count = 0
    processed_imgs = os.listdir(dir)
    total_imgs = len(processed_imgs)  
    t1 = 70
    t2 = 150
    false_positive_threshold = 70 # usually ranges from 20 - 100
    for img_file in processed_imgs:
        if count % 100 == 0:
            print(f"Extracting shapes for {count}/{total_imgs} images")
        try:
            # Read in preprocessed images
            img_path = os.path.join(dir, img_file)
            image = io.imread(img_path)
            # blur image further for ease of hough transform
            image = cv2.GaussianBlur(image, (5, 5), 2)
            # Extract the image dimensions and hough circles
            height, width = image.shape[:2] 
            circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20, 
                                       param1=t1, param2=false_positive_threshold, minRadius=int(width * .20), maxRadius=0)

            # if there were circles detected, 
            if circles is not None:
                circles = np.uint16(np.around(circles))  # round the circle coordinates
                mask = np.zeros_like(image)  # create black mask
                for i in circles[0, :]:
                    center = (i[0], i[1])  
                    radius = i[2]  
                    cv2.circle(mask, center, radius, 255, thickness=-1)  # draw the filled circle on the mask
                
                # Apply the mask to the original image
                segmented_image = cv2.bitwise_and(image, image, mask=mask)
                # Save the segmented image
                base_name = os.path.splitext(img_file)[0]
                os.makedirs("segmented_data", exist_ok=True)
                output_path = os.path.join("segmented_data/", f"{base_name}_segmented.jpg")
                cv2.imwrite(output_path, segmented_image)

                os.makedirs("masks", exist_ok=True)
                output_path = os.path.join("masks/", f"{base_name}_mask.jpg")
                cv2.imwrite(output_path, mask)

                #cv2.imshow("Segmented Traffic Sign", segmented_image)
                #cv2.waitKey(0)
            #else:
                #cv2.imshow("NO DETECTED traffic Sign", image)
                #cv2.waitKey(0)
                
            count += 1
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
    return circles

def number_features(dir):
    # init EasyOCR reader
    reader = easyocr.Reader(['en'])  # 'en' for English
    # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, or S for STOP
    allowed_labels = ["20", "30", "50", "60", "70", "STOP"] 
    all_feature_vectors = []
    image_files = os.listdir(dir)
    count = 0
    num_images = len(image_files)
    for img_file in image_files:
        if count % 100 == 0:
            print(f"Detecting numbers in {count}/{num_images} images")
        count += 1
        img_path = os.path.join(dir, img_file)
        try:   
            feature_vector = [] 
            results = reader.readtext(img_path)
            #for (bbox, text, prob) in results:
                #print(f"Detected: {text} (Confidence: {prob:.2f})")
            # detect only permitted characters
            detected_chars = set(text.upper() for (_, text, _) in results if text.upper() in allowed_labels)
            # Convert detected items into a one 11D vector
            feature_vector = [1 if label in detected_chars else 0 for label in allowed_labels]
            all_feature_vectors.append(feature_vector)
            #print(feature_vector)
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
    return all_feature_vectors

def predict_cluster(image_path, kmeans_model):
    image = io.imread(image_path)

    # Extract HOG features
    features = hog(image, orientations=9, pixels_per_cell=(8, 8), 
                   cells_per_block=(2, 2), visualize=False)
    
    # Ensure features have the correct shape for computation
    if features is not None:
        features = features.reshape(1, -1)  # Reshape to (1, n_features)
        
        # Get cluster centers from the trained k-means model
        cluster_centers = kmeans_model.cluster_centers_  # Shape: (n_clusters, n_features)
        
        # Compute Euclidean distances between the features and each cluster center
        distances = np.linalg.norm(cluster_centers - features, axis=1)  # Shape: (n_clusters,)
        
        # Find the closest cluster (minimum distance)
        closest_cluster = np.argmin(distances)
        
        print(f"The image belongs to cluster: {closest_cluster} (Min Distance: {distances[closest_cluster]:.4f})")
        return closest_cluster
    else:
        print("Feature extraction failed.")
        return None

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
        if count >= 1000:
            break
            
        if count % 100 == 0:
            print(f"At {count}/{len(processed_imgs)} images")
            
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

def read_label_file(label_path):
    with open(label_path, 'r') as f:
        line = f.readline().strip()
        parts = line.split()
        if len(parts) >= 1:
            return parts[0]
    return None

def load_features_with_labels(segmented_dir, original_image_dir, label_dir):
    feature_vectors = []
    true_labels = []
    count = 0
    
    segmented_files = [f for f in os.listdir(segmented_dir) if f.endswith(('.jpg', '.png'))]
    
    for segmented_file in segmented_files:
        if count >= 1000:
            break
            
        if count % 100 == 0:
            print(f"Processing {count}/{len(segmented_files)} images")
        base_name = os.path.splitext(segmented_file)[0]
        if base_name.endswith("_segmented"):
            original_base = base_name.replace("_segmented", "")
        else:
            original_base = base_name

        original_file = None
        for ext in ['.jpg', '.png']:
            potential_file = original_base + ext
            if os.path.exists(os.path.join(original_image_dir, potential_file)):
                original_file = potential_file
                break
        
        if original_file is None:
            print(f"Could not find original image for {segmented_file}")
            continue
        
        # Get the label file path based on the original image name
        label_base = os.path.splitext(original_file)[0]
        label_path = os.path.join(label_dir, f"{label_base}.txt")
        
        if not os.path.exists(label_path):
            print(f"Label file not found for {original_file}")
            continue
        
        try:
            # Read segmented image and extract features
            segmented_path = os.path.join(segmented_dir, segmented_file)
            image = io.imread(segmented_path)
            features = hog(image, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=False)
            class_id = read_label_file(label_path)
            if class_id is None:
                print(f"Could not parse class ID from {label_path}")
                continue
            
            feature_vectors.append(features)
            true_labels.append(class_id)
            count += 1
        except Exception as e:
            print(f"Error processing {segmented_file}: {str(e)}")
    
    if len(feature_vectors) == 0:
        print("No valid data found with labels!")
        return np.array([]), np.array([])
    
    return np.array(feature_vectors), np.array(true_labels)

def map_clusters_to_classes(cluster_labels, true_labels):
    class_names = ['Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100',
                   'Speed Limit 110', 'Speed Limit 120', 'Speed Limit 20',
                   'Speed Limit 30', 'Speed Limit 40', 'Speed Limit 50',
                   'Speed Limit 60', 'Speed Limit 70', 'Speed Limit 80', 'Speed Limit 90', 'Stop']
    cluster_to_class = {}
    cluster_to_name = {}
    
    for cluster_id in np.unique(cluster_labels):
        cluster_true_labels = true_labels[cluster_labels == cluster_id]
        unique_labels, counts = np.unique(cluster_true_labels, return_counts=True)
        most_common_idx = np.argmax(counts)
        most_common_label = unique_labels[most_common_idx]
        try:
            label_idx = int(most_common_label)
            class_name = class_names[label_idx] if label_idx < len(class_names) else f"Unknown Class {most_common_label}"
        except (ValueError, IndexError):
            class_name = f"Class {most_common_label}"
        
        cluster_to_class[cluster_id] = most_common_label
        cluster_to_name[cluster_id] = class_name
        
        print(f"Cluster {cluster_id} -> {class_name} (ID: {most_common_label}, {counts[most_common_idx]}/{len(cluster_true_labels)} images)")
    
    return cluster_to_class, cluster_to_name

def predict_class(image_path, kmeans_model, cluster_to_class, cluster_to_name):
    image = io.imread(image_path)

    features = hog(image, orientations=9, pixels_per_cell=(8, 8), 
                  cells_per_block=(2, 2), visualize=False)
    
    if features is not None:
        features = features.reshape(1, -1)
        cluster_centers = kmeans_model.cluster_centers_
        distances = np.linalg.norm(cluster_centers - features, axis=1)
        closest_cluster = np.argmin(distances)
        
        # Map cluster to class and name
        if closest_cluster in cluster_to_class:
            predicted_class = cluster_to_class[closest_cluster]
            predicted_name = cluster_to_name[closest_cluster]
            print(f"Image {os.path.basename(image_path)} belongs to: {predicted_name} (Cluster: {closest_cluster}, Distance: {distances[closest_cluster]:.4f})")
            return predicted_class, predicted_name
        else:
            print(f"No class mapping for cluster {closest_cluster}")
            return None, None
    else:
        print("Feature extraction failed")
        return None, None
    
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python preprocess.py image_dir output_dir label_dir")
        exit(1)
    
    image_dir = sys.argv[1]
    output_dir = sys.argv[2]
    label_dir = sys.argv[3]

    preprocess(image_dir, output_dir)
    # creates masks and writes to an output directory
    segment_images(output_dir)
    nums = number_features("./segmented_data")
    colors = red.load_corresponding_colors(image_dir, "./masks")

    feature_vectors, true_labels = load_features_with_labels("segmented_data", image_dir, label_dir)

    n_clusters = min(10, len(feature_vectors))
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans_model.fit_predict(feature_vectors)

    cluster_to_class, cluster_to_name = map_clusters_to_classes(cluster_labels, true_labels)
    test_images = [
        './output_dir/00000_00000_00004_png.rf.8737f80bd4f1455970179b3df433fba5.jpg',
        './output_dir/00000_00004_00019_png.rf.e587d94b21592a7cce2cdba1b9e0c2b8.jpg',
        './output_dir/road665_png.rf.853969c1e3fa5fac142f8e7852819a09.jpg'
    ]
    
    for test_image in test_images:
        if os.path.exists(test_image):
            predict_class(test_image, kmeans_model, cluster_to_class, cluster_to_name)
        else:
            print(f"Test image not found: {test_image}")