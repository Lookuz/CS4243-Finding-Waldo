import os
import cv2 as cv
import numpy as np
import cyvlfeat as vlfeat

from utils import *

vocab_step = 5
vocab_size = 4
sample_step = 3
sample_size = 4


def get_color_feat(im):
    hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[:, :, 0]
    hist = cv.calcHist(hsv_im, channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    hist /= np.linalg(hist)
    return hist.reshape(-1)


def get_contour_feat(im, step=3, size=4):
    grey_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, descriptors = vlfeat.sift.dsift(grey_im, fast=True, step=step, size=size)
    return descriptors

def get_contour_feat_vocab(im):
    return get_contour_feat(im, vocab_step, vocab_size)

def get_contour_feat_sample(im):
    return get_contour_feat(im, sample_step, sample_size)


def positive_source(name):
    goal_dir = os.getcwd()
    img_src = os.path.join(goal_dir, 'datasets', 'positives', name)
    if name == 'waldo':
        srcs = []
        for dir in load_full_subdir(img_src):
            srcs.extend(load_full_subdir(dir))
        return srcs
    else:
        return load_full_subdir(img_src)


def potential_source(name):
    goal_dir = os.getcwd()
    img_src = os.path.join(goal_dir, 'datasets', 'potential')
    if name == 'wizard':
        return load_full_subdir(os.path.join(img_src, 'wizard'))
    else:
        return load_full_subdir(os.path.join(img_src, 'waldo_wenda'))


def load_positive_patch(name):
    paths = positive_source(name)
    for img_path in paths:
        yield np.load(img_path)


def load_sub_positive_patch(name):
    paths = positive_source(name)
    paths.extend(potential_source(name))
    for img_path in paths:
        yield np.load(img_path)


def extract_vocabs(name, sub_correct=False):
    num_vocab_per_patch = 20
    vocab_size = 100
    sifts = []
    loader = load_sub_positive_patch if sub_correct else load_positive_patch()

    for img_data in loader(name):
        descriptors = get_contour_feat_vocab(img_data)
        all_idxs = np.arange(len(descriptors))
        np.random.shuffle(all_idxs)
        sifts.extend(descriptors[:num_vocab_per_patch])

    vocabs = vlfeat.kmeans.kmeans(np.array(sifts, dtype='f'), vocab_size)
    return vocabs


def extract_color_features(name, sub_correct=False):
    color_feats = np.zeros(256)
    loader = load_sub_positive_patch if sub_correct else load_positive_patch

    for img_data in loader(name):
        color_feats += get_color_feat(img_data)

    color_feats = color_feats / np.linalg.norm(color_feats)
    return color_feats

