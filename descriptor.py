"""
    Descriptors Methods used for feature extraction.
    Notice the image passed are raw cv.imread result,
    i.e. haven't divided by 255, still in BGR order
"""

import numpy as np
import cv2 as cv
import cyvlfeat as vlfeat

def color(im, **kwargs):
    assert im.ndim == 3
    hsv_im = cv.cvtColor(im, cv.COLOR_BGR2HSV)[:, :, 0]
    hist = cv.calcHist(hsv_im, channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    hist /= np.linalg(hist)
    return hist.reshape(-1)

def sift(im, **kwargs):
    step = kwargs.get('step', 3)
    size = kwargs.get('size', 4)
    if im.ndim == 3:
        im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    _, descriptors = vlfeat.sift.dsift(im, fast=True, step=step, size=size)
    return descriptors

def surf(im, **kwargs):
    DSP_OBJ = kwargs.get('DSP_OBJ')
    _, descriptors = DSP_OBJ.detectAndCompute(im, None)
    return descriptors


def akaze(im, **kwargs):
    DSP_OBJ = kwargs.get('DSP_OBJ')
    _, descriptors = DSP_OBJ.detectAndCompute(im, None)
    return descriptors
