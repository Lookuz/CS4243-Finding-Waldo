import os

from dataset import *
from descriptor import *

vocab_step = 5
vocab_size = 4
sample_step = 3
sample_size = 4

desciptor_map = {
    'color': color,
    'sift': sift
}


def get_feat(img, **kwargs):
    method = kwargs.get('method', 'sift')
    return desciptor_map[method](img, **kwargs)


def extract_vocabs(dataloader, method='sift', **kwargs):
    num_vocab_per_patch = 20
    vocab_num = 100
    feats = []

    for img_data in dataloader:
        descriptors = get_feat(img_data, method=method, **kwargs)
        all_idxs = np.arange(len(descriptors))
        np.random.shuffle(all_idxs)
        feats.extend(descriptors[:num_vocab_per_patch])

    vocabs = vlfeat.kmeans.kmeans(np.array(feats, dtype='f'), vocab_num)
    return vocabs


def extract_color_features(dataloader):
    color_feats = np.zeros(256)

    for img_data in dataloader:
        color_feats += get_feat(img_data, method='color')

    color_feats = color_feats / np.linalg.norm(color_feats)
    return color_feats
