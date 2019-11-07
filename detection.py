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
    'waldo' : os.path.join(classifiers_filepath, 'waldo/')
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

# Function that loads all classifiers for the given detection class
def load_classifiers(detection_class):
    classifiers = []
    
    for path in load_full_subdir(classifiers_filepath_map[detection_class]):
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

# TODO: Add functions to perform detection on validation set of images