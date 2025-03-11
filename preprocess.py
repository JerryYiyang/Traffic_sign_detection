from skimage import io, color, filters, util
import os
import sys


def preprocess(image_dir, output_dir):
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
                output_path = os.path.join(output_dir, f"{base_name}.png")
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