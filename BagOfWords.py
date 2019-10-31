import cv2
import numpy as np
import scipy
import random
import os
import matplotlib.pyplot as plt
import cyvlfeat as vlfeat

from scipy.spatial.distance import cdist

KAZE_DESC_SIZE = 64

# Function that extracts features as descriptors from the given image 
# TODO: Currently uses KAZE descriptor. To add SIFT/SURF descriptor functionality
def extract_feature(image, vector_size=64, limit=True):

    kaze = cv2.KAZE_create()
    # Identify keypoints
    keypoints = kaze.detect(image)
    # Take only the the most significant vector_size keypoints
    keypoints = sorted(keypoints, key=lambda x: -x.response) # Larger response in front
    keypoints = keypoints[:vector_size]
    # Take descriptor around window of keypoints
    keypoints, descriptor = kaze.compute(image, keypoints)
    # Handle missing keypoints
    if descriptor is None or keypoints is None:
        return None, None

    return keypoints, descriptor

# Extracts feature descriptors from a list of images. 
# Runs the extract_feature function as a subroutine
def extract_features(images, vector_size=64, mode='RGB'):
    descriptors = np.array([]).reshape(0, vector_size)
    for image in images:
        keypoints, descriptor = extract_feature(image, vector_size=vector_size)
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
def extract_histograms(images, bag_of_words, metric='euclidean'):
    vocab_size = bag_of_words.shape[0]
    histograms = []
    
    for image in images:
        _, descriptors = extract_feature(image, limit=False)
        if descriptors is None:
            histogram = np.zeros(vocab_size)
        else:
            histogram = extract_histogram(descriptors, bag_of_words, metric=metric)
            
        histograms.append(histogram)

    return np.array(histograms)