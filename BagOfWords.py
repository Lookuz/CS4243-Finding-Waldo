import cv2
import numpy as np
import scipy
import random
import os
import matplotlib.pyplot as plt
import cyvlfeat as vlfeat

from scipy.spatial.distance import cdist

from SlidingWindow import *

descriptor_sizes = {
    'kaze' : 64,
    'sift' : 128,
    'brisk': 64
}

# Function that extracts features as descriptors from the given image 
# Set the limit parameter to false to take the full range of descriptors in the image
# step and size parameters are used for SIFT descriptor
def extract_feature(image, vector_size=64, step=3, size=4, desc_type='kaze'):
    if desc_type == 'kaze':
        # kaze = cv2.KAZE_create()
        kaze = cv2.AKAZE_create()
        # Compute keypoints and descriptors
        keypoints, descriptor = kaze.detectAndCompute(image, None)
    elif desc_type == 'sift':
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints, descriptor = sift.detectAndCompute(image, None)
    elif desc_type == 'brisk':
        brisk = cv2.BRISK_create()
        keypoints, descriptor = brisk.detectAndCompute(image, None)

    # Handle missing keypoints
    if descriptor is None or keypoints is None:
        return None, None

    return keypoints, descriptor

# Extracts feature descriptors from a list of images. 
# Runs the extract_feature function as a subroutine
def extract_features(images, mode='RGB', desc_type='kaze'):
    vector_size = descriptor_sizes[desc_type]
    descriptors = np.array([]).reshape(0, vector_size)
    for image in images:
        keypoints, descriptor = extract_feature(image, vector_size=vector_size, desc_type=desc_type)
        if descriptor is not None:
            descriptors = np.concatenate((descriptors, descriptor), axis=0)
    
    return descriptors

# Function that clusters the feature descriptors to obtain 
# The cluster centers as visual words
def cluster_features(features, num_clusters=50):
    bag_of_words = vlfeat.kmeans.kmeans(features, num_clusters)
    return bag_of_words

# Extracts feature historam from a given descriptor
# Bins the descriptors to nearest visual word in the BOW provided
# TODO: Try other distance metrics: chi-squared, bhattacharyya
def extract_histogram(descriptors, bag_of_words, metric='euclidean'):
    vocab_size = bag_of_words.shape[0]
    
    distances = cdist(descriptors, bag_of_words, metric)
    word_indices = np.argmin(distances, axis=1)
    histogram = np.bincount(word_indices, minlength=vocab_size)
    histogram = histogram / np.linalg.norm(histogram) # Normalization

    return histogram

# Function that extracts the binned histograms using the given bag of words
def extract_histograms(images, bag_of_words, metric='euclidean', desc_type='kaze'):
    vocab_size = bag_of_words.shape[0]
    histograms = []
    
    for image in images:
        _, descriptors = extract_feature(image, desc_type=desc_type)
        if descriptors is None:
            histogram = np.zeros(vocab_size)
        else:
            histogram = extract_histogram(descriptors, bag_of_words, metric=metric)
            
        histograms.append(histogram)

    return np.array(histograms)

# Functions that detects Waldo in the given image using window-based techniques
# Performs a sliding window over each window on the image with size specified by window_size,
# and scores each window using the model supplied
# Pyramidal scaling is also applied to apply sliding window over multiscale situations
# window_size = (r, c)/ (y, x)
def detect(image, bag_of_words, clf, step_size=250, window_size=(400, 200), scale=1.5, desc_type='kaze'):
    detections = [] # To store detected window coordinates
    current_scale = 0
    pyramid_window = (image.shape[1] // 4, image.shape[0] // 4)
    
    # Apply pyramidal sliding window
    for scaled_image in image_pyramid(image, scale=scale, minSize=pyramid_window):
        # Resized image too small
        if scaled_image.shape[0] < pyramid_window[1] or scaled_image.shape[1] < pyramid_window[0]:
            break
        
        for (coordinates, window) in sliding_window(scaled_image, step_size=step_size, window_size=window_size):
            # if window.shape[0] != window_size[0] or window.shape[1] != window_size[1]:
            #         continue
            y, x, y_end, x_end = coordinates

            # For this window, extract the descriptors then extract the histogram vector from descriptors
            # _, descriptors = extract_feature(window, limit=False)
            # feature_vector = extract_histogram(descriptors, bag_of_words)
            feature_vector = extract_histograms([window], bag_of_words, desc_type=desc_type)
            # Predict Waldo
            prediction = clf.predict(feature_vector)[0]
            predict_score = clf.predict_proba(feature_vector)[0][1] # Get prediction probability
            if prediction == 1:
                # Rescale coordinates
                win_x = int(x * (scale ** current_scale))
                win_y = int(y * (scale ** current_scale))
                win_x_end = win_x + int(window_size[1] * (scale ** current_scale))
                win_x_end = min(win_x_end, image.shape[1])
                win_y_end = win_y + int(window_size[0] * (scale ** current_scale))
                win_y_end = min(win_y_end, image.shape[0])
                # Add bounding box
                detections.append((win_x, win_y, win_x_end, win_y_end, predict_score))
                
        current_scale += 1
    
    # Perform Non-Maximum Suppression
    detections = non_max_suppression(detections, threshold=0.3)
    
    return detections