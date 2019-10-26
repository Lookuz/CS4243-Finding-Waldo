# Utilities file that contains operations supporting image processing
import numpy as np
import matplotlib.pyplot as plt 
import os.path as osp
from skimage import io

# Standard 3x3 sobel filters
sobel_horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32) # for change in the x-direction
sobel_vertical = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32) # for change in the y-direction

# Loads an image from the given file path if the file path is valid
# Returns the image in numpy.array format containing 3 RGB channels
def load_image(file_name):
    if not osp.exists(file_name):
        print('{} does not exist'.format(file_name))
        return
    image = io.imread(file_name)
    return np.array(image)

# Saves the image to the given file path
def save_image(image, file_path):
    io.imsave(file_path, image)

# Shows the image in np.array format provided
def show_image(image):
    plt.imshow(image)
    plt.show()

def rgb_to_gray(image, channels=None):
    if (channels is not None) and len(channels) != 3:
        print('Channels provided should be in the form: [R, G, B]') 
        return

    if len(image.shape) != 3 or image.shape[2] != 3:
        print('Image should have 3 RGB channels')
        return

    if channels is None:
        rgb_weights = [0.299, 0.587, 0.114]
    
    greyscale_img = np.dot(image, rgb_weights) # Perform dot product of RGB channels on the RGB weights for each pixel

    return greyscale_img/255.

# Generates a gaussian kernel of size kxk
# Size provided must be an odd size
# NOTE: scipy library provides a gaussian kernel filtering function: scipy.ndimage.filters.gaussian_filter
def gaussian_kernel(size, sigma=1):
    if (int(size) <= 0) or (int(size) % 2 == 0):
        print('Gaussian Kernel size must be odd and greater than 0')
        return

    size = int(size)//2
    x, y = np.mgrid[-size:size+1, -size:size+1] # Generate distances from center
    normal = 1 / (2.0 * np.pi * (sigma ** 2))
    gaussian = np.exp(-((x**2 + y**2) / (2.0 * sigma * 2))) * normal

    return gaussian

# def convolve():