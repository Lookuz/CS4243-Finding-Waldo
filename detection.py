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
    except OSError as e:
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

# Similar to convert_bboxes_to_file but to be used for a list of images' bounding boxes
def convert_imgs_bboxes_to_file(img_names, img_bboxes, classname):
    """
    img_names: list of image names
    img_bboxes: list of list of bounding boxes (x1, x2, y1, y2, score)
    classname: name of the class
    """
    
    curr_wd = os.getcwd()
    output_bboxes_path = os.path.join(curr_wd, 'baseline', 'output')
    bboxes_file_path = os.path.join(output_bboxes_path, classname + ".txt")
    
    if not os.path.exists(output_bboxes_path):
        os.makedirs(output_bboxes_path)
    
    with open(bboxes_file_path, 'w') as fp:
        for idx, imgname in enumerate(img_names) :
            convert_bboxes_to_file(imgname, img_bboxes[idx], classname, reset_file=False)

# Function that prepare a list of bounding boxes as a text file to be fed to voc_eval
# File is created anew by default, set reset_file to False to append to an existing file
# Used for one image
def convert_bboxes_to_file(imgname, bboxes, classname, reset_file=True):
    """
    imgname: name of the image
    bboxes: list of bounding boxes (x1, x2, y1, y2, score)
    classname: name of the class
    reset_file: whether to reset the file if it alr exists
    """
    
    curr_wd = os.getcwd()
    output_bboxes_path = os.path.join(curr_wd, 'baseline', 'output')
    bboxes_file_path = os.path.join(output_bboxes_path, classname + ".txt")
    
    if not os.path.exists(output_bboxes_path):
        os.makedirs(output_bboxes_path)
        
    open_mode = 'w'
    if not reset_file:
        open_mode = 'a'
    
    with open(bboxes_file_path, open_mode) as fp:
        for box in bboxes:
            fp.write(('{} {} {} {} {} {}\n'.format(imgname, box[4], box[0], box[1], box[2], box[3])))

# Function that performs the final object detection on a list of given images
# Saves the detections to the baseline folder 
"""
Parameters: 
image_filepath - List of file paths to the images to be processed
classifiers - Classification models to be used for scoring windows
classname - Type of object to be detected: waldo, wenda or wizard
detection_type - Determines what type of detection to use:
    window_scoring - uses sliding window object detection
    haar_filtering - combines sliding window objection with haar cascade filtering
    haar_detection - performs haar cascade detection on the image, then score each haar detection with classifiers
"""
def object_detection(image_filepath, classname, desc_type='sift', detection_type='window_scoring', bovw=None, classifiers=None):

    # Load default bovw and classifiers
    if bovw is None:
        bovw = load_bovw(classname, desc_type=desc_type)
    # NOTE: Feel free to use optional classifiers argument to specify own classifiers
    if classifiers is None:
        classifiers = load_classifiers(classname)
    
    cascade_classifier = load_haar_classifier() # Only for Waldo

    img_bboxes = [] # List of list of bounding boxes. Index of the bounding boxes is equivalent to index of image name in image_list
    img_names = [os.path.splitext(os.path.basename(path))[0] for path in image_filepath] # Extract image names
    
    # For each image:
    for path in image_filepath:
        print('Processing Image ', path)
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        # obtain detections using the specified detection_type method
        if detection_type == 'window_scoring' or classname in ['wenda', 'wizard']:
            detections = detect(image, bovw, classifiers, window_scale=5, desc_type=desc_type, suppress=True)
        elif detection_type == 'haar_filtering':
            detections = detect(image, bovw, classifiers, window_scale=5, desc_type=desc_type, suppress=True)
            detections = haar_filtering(image, classifiers, cascade_classifier)
        elif detection_type == 'haar_detection':
            detections = haar_detection(image, cascade_classifier, classifiers, bovw)
            detections = non_max_suppression(detections, threshold=0.1, score_threshold=0.5)
        else: 
            print('Invalid detection type!')
            return

        # Append the detections to img_bboxes
        img_bboxes.append(detections)
    
    # Save detections
    convert_imgs_bboxes_to_file(img_names, img_bboxes, classname)

    return img_bboxes
