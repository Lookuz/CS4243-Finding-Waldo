"""
    Python module that consolidates detection algorithms to perform object detection
"""
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import cv2
import os
import time
from sklearn.utils import shuffle
from sklearn import metrics as sk_metrics
import pickle as pkl
import scipy.ndimage as ndimage

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost

from dataset import *
from descriptor import *
from SlidingWindow import *
from BagOfWords import *

detection_classes = ['waldo'] # TODO: Add Wenda and Wizard once detection classifiers and bovw have been trained for them

"""
File Paths
"""
curr_wd = os.getcwd()
classifiers_filepath = os.path.join(curr_wd, 'models/')
classifiers_filepath_map = {
    'waldo' : {
        'sift' : os.path.join(classifiers_filepath, 'waldo/', 'sift/'),
        'kaze' : os.path.join(classifiers_filepath, 'waldo/', 'kaze/')
    }
}

haar_classifier_filepath_map = {
    'waldo' : os.path.join(classifiers_filepath, 'waldo/', 'haar_cascade.xml')
}


bovw_filepath = os.path.join(curr_wd, 'bovw/')
bovw_filepath_waldo = os.path.join(bovw_filepath, 'waldo/')
bovw_filepath_map = {
    'waldo' : {
        'sift' : os.path.join(bovw_filepath_waldo, 'bovw_sift_80.pkl'),
        'kaze' : os.path.join(bovw_filepath_waldo, 'bovw_kaze_80.pkl'),
        'brisk' : os.path.join(bovw_filepath_waldo, 'bovw_brisk_80.pkl')
    }
}


# Loads pre-trained haar cascade classifier 
# Uses XML file from the defined file path to load the trained parameters
def load_haar_classifier(detection_class='waldo'):
    try:
        haar_classifier = cv2.CascadeClassifier(haar_classifier_filepath_map[detection_class])
        return haar_classifier
    except Error as e:
        print(e)

# Function that loads all classifiers for the given detection class
def load_classifiers(detection_class, desc_type='sift'):
    classifiers = []
    
    for path in load_full_subdir(classifiers_filepath_map[detection_class][desc_type]):
        with open(path, 'rb') as f:
            if path.endswith('.pkl'):
                classifiers.append(pickle.load(f))
    
    return classifiers

# Function that loads the bag of visual words vocabulary 
# For the provided detection class using the desc_type descriptor type
def load_bovw(detection_class, desc_type='sift'):
    path = bovw_filepath_map[detection_class][desc_type]

    with open(path, 'rb') as f:
        bovw = pickle.load(f)
    
    return bovw

# Perform haar cascade filtering on the sliding window detection bounding boxes
# detections: bounding boxes from sliding window detections. Should be in the format (x1, y1, x2, y2)
# which are the coordinates of the four corners of the box
# cascade_classifer: haar cascade classifier to be used to identify facial features within detections
# Use the load_haar_classifier function to load pre-trained haar classifier
def haar_filtering(image, detections, cascade_classifier):
    final_detections = []
    for (x1, y1, x2, y2, score) in detections:
        patch = image[y1:y2, x1:x2]
        # NOTE: Feel free to modifiy these parameters
        # scaleFactor: For multiscale detection. Higher scale is faster detection, but less accurate. Recommend adjustments in intervals of 0.05
        # minNeighbours: Number of nieghbours for KNN classification. Higher value means greater threshold, but may result in missed detections
        # minSize/maxSize: range of size of patches
        rectangles = cascade_classifier.detectMultiScale(patch, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100), maxSize=(400, 400))

        if len(rectangles) > 0:
            final_detections.append((x1, y1, x2, y2, score))
    
    return final_detections

# Obtains haar detection boxes from image, then score the haar detections
# and select only the detections that score beyond the specified threshold

def haar_detection(image, cascade_classifier, classifiers, bovw, threshold=0.5):
    # Get haar detections
    haar_boxes = cascade_classifier.detectMultiScale(image, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100), maxSize=(400, 400))
    
    detections = []
    # Score each haar detection
    for box in haar_boxes:
        x = box[0]
        y = box[1]
        x_end = x + box[2]
        y_end = y + box[3]
        patch = image[y:y_end, x:x_end]
        
        feature_vector = extract_histograms([patch], bovw, desc_type='sift')
        prediction, predict_score = get_prediction(classifiers, feature_vector)
        
        # Threshold the score
        if predict_score >= threshold:
            detections.append((x, y, x_end, y_end, predict_score))
    
    return detections
