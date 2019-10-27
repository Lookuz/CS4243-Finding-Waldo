"""
Image Segmentation using Clustering
Author: Lukaz
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
from sklearn.cluster import KMeans

# NOTE: Segmentation Procedure 
# Run the generate_kmeans_clusters function to get the kmeans_model
# Run the segment_image using the kmeans_model to get the segmented image

# Function that generates the clusters using K-Means Clustering
# Returns the kmeans object to be used for future predictions
def generate_kmeans_clusters(template_image_path, n_clusters=5, random_state=0):
    # Read image from file path
    if os.path.exists(template_image_path):
        template_image = plt.imread(template_image_path)
        template_image = template_image/255. # Normalize
    else:
        print('File path does exist')
        return

    # Vectorize template
    template_vector = template_image.reshape((-1, 3))
    template_vector = np.float32(template_vector)

    # Initialize KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans = kmeans.fit(template_vector) # Cluster on template

    return kmeans

# Segments a given image using the specified model
# To be used in conjunction with generate_kmean_clusters,
# using the kmeans_model generated by it as the model
def segment_image(image, model):
    image = image/255. # Normalize
    image_vector = image.reshape((-1, 3)) # Transform as feature vector
    image_vector = np.float32(image_vector)

    labels = model.predict(image_vector) # Get labels for each pixel
    segmented_image = model.cluster_centers_[labels]
    segmented_image = segmented_image.reshape((image.shape))

    return segmented_image

def segment_images(image_list, model):
    return [lambda x: segment_image(x, model) for x in image_list]
