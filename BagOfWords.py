import cv2
import numpy as np
import scipy
import random
import os
import matplotlib.pyplot as plt
import cyvlfeat as vlfeat
from sklearn import metrics as sk_metrics
from sklearn.utils import shuffle

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from scipy.spatial.distance import cdist

from SlidingWindow import *

descriptor_sizes = {
    'kaze' : 61,
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

# Function that conslidates predictions from the model, or an ensemble of models
# Uses a simple bagging structure for outputting the conslidated prediction
def get_prediction(clf, feature_vector):
    if type(clf) == list:
        prediction = clf.predict(feature_vector)[0]
        predict_score = clf.predict_proba(feature_vector)[0][1]
    else:
        predict_scores = predict_scores = [x.predict_proba(feature_vector)[0][1] for x in clf]
        predict_score = sum(predict_scores) / len(predict_scores)
        prediction = 1.0 if predict_score > 0.5 else 0.0

    return prediction, predict_score


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


# detect that need no vocabs provided
def detect_with_clf(image, clf, step_size=250, window_size=(400, 200), scale=1.5, pyramid_window=(2000, 2000)):
    detections = []  # To store detected window coordinates
    current_scale = 0

    # Apply pyramidal sliding window
    for scaled_image in image_pyramid(image, scale=scale, minSize=pyramid_window):
        # Resized image too small
        if scaled_image.shape[0] < pyramid_window[1] or scaled_image.shape[1] < pyramid_window[0]:
            break

        for (coordinates, window) in sliding_window(scaled_image, step_size=step_size, window_size=window_size):
            # if window.shape[0] != window_size[0] or window.shape[1] != window_size[1]:
            #         continue
            y, x, y_end, x_end = coordinates
            predict_score = clf.predict_proba([window])[0][1]  # Get prediction probability
            if predict_score > 0.5:
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
    
    return detections


# Function that evaluates the given classifier clf on the validation dataset provided
def evaluate_classifier(clf, val, val_labels):
    val_predict = clf.predict(val)
    precision = sk_metrics.precision_score(y_true=val_labels, y_pred=val_predict)
    recall = sk_metrics.recall_score(y_true=val_labels, y_pred=val_predict)
    f1_score = sk_metrics.f1_score(y_true=val_labels, y_pred=val_predict)
    print('Precision: %.3f\nRecall: %.3f\nF1 Score: %.3f' % (precision, recall, f1_score))

# Function that trains a set of models on the training data provided
# And tests that on the validation dataset provided
# Returns tuple of models fit on the training dataset
# train: training images provided alongside the labels
# val: validation set to evalute the performance of trained models
def generate_models(train_x, train_y, val_x, val_y, bag_of_words, model='ensemble', desc_type='kaze'):
    # Extract histogram vectors from data set
    train_histograms = extract_histograms(train_x, bag_of_words, desc_type=desc_type)
    val_histograms = extract_histograms(val_x, bag_of_words, desc_type=desc_type)
    # Shuffle validation and training set
    train_histograms, train_y = shuffle(train_histograms, train_y)
    val_histograms, val_y = shuffle(val_histograms, val_y)

    models = []

    if model == 'svm' or model == 'ensemble':
        svm_linear = SVC(kernel='linear', C=0.5, probability=True)
        svm_linear.fit(train_histograms, train_y)
        print('Performance of Linear SVM on validation set:')
        evaluate_classifier(svm_linear, val_histograms, val_y)
        models.append(svm_linear)

        svm_rbf = SVC(kernel='rbf', probability=True)
        svm_rbf.fit(train_histograms, train_y)
        print('Performance of RBF SVM on validation set:')
        evaluate_classifier(svm_rbf, val_histograms, val_y)
        models.append(svm_rbf)
    
    if model == 'gbc' or model == 'ensemble':
        gbc = GradientBoostingClassifier()
        gbc.fit(train_histograms, train_y)
        print('Performance of GradientBoostingClassifier on validation set:')
        evaluate_classifier(gbc, val_histograms, val_y)
        models.append(gbc)
    
    if model == 'rf' or model == 'ensemble':
        rf = RandomForestClassifier()
        rf.fit(train_histograms, train_y)
        print('Performance of RandomForestClassifier on validation set:')
        evaluate_classifier(rf, val_histograms, val_y)
        models.append(rf)

    return models[0] if len(models) == 1 else tuple(models)