#pip install scikit-image
from skimage import color
from skimage.feature import hog
from skimage import data, exposure, io
import matplotlib.pyplot as plt

# USED GEEKS FOR GEEKS AS CODE REFERENCE FOR HOG
# Loading an example image
image = io.imread("./car/train/images/00000_00000_00025_png.rf.1fccf78acacf5d504932d3ac185e6f5e.jpg")
image_gray = color.rgb2gray(image) # Converting image to grayscale

# Extract HOG features
features, hog_image = hog(image_gray, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=True)
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image_gray, cmap='gray')
plt.title('Input image')

plt.subplot(1, 2, 2)
plt.imshow(hog_image, cmap='gray')
plt.title('HOG features')
plt.show()

#Name of Classes: 
#   Green Light, 
#   Red Light, 
#   Speed Limit 10, 
#   Speed Limit 100, 
#   Speed Limit 110, 
#   Speed Limit 120, 
#   Speed Limit 20, 
#   Speed Limit 30, 
#   Speed Limit 40, 
#   Speed Limit 50, 
#   Speed Limit 60, 
#   Speed Limit 70, 
#   Speed Limit 80, 
#   Speed Limit 90, 
#   Stop