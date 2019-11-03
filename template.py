import os

from dataset import *
from descriptor import *

vocab_step = 5
vocab_size = 4
sample_step = 3
sample_size = 4

desciptor_map = {
    'color': color,
    'sift': sift,
    'surf': surf,
    'akaze': akaze
}


def get_feat(img, **kwargs):
    method = kwargs.get('method', 'sift')
    return desciptor_map[method](img, **kwargs)


def extract_vocabs(dataloader, method='sift', **kwargs):
    num_vocab_per_patch = 20
    vocab_num = 100
    feats = []

    for img_data, gt in dataloader:
        if gt == 0:
            continue
        descriptors = get_feat(img_data, method=method, **kwargs)
        if descriptors is None:
            print('one image get empty descriptors for vocab')
            continue
        all_idxs = np.arange(len(descriptors))
        np.random.shuffle(all_idxs)
        if len(descriptors) > num_vocab_per_patch:
            feats.extend(descriptors[:num_vocab_per_patch])
        else:
            contrib_num = len(descriptors) // 2
            feats.extend(descriptors[:contrib_num])

    vocabs = vlfeat.kmeans.kmeans(np.array(feats, dtype='f'), vocab_num)
    return vocabs


def extract_color_features(dataloader):
    color_feats = np.zeros(256)

    for img_data, gt in dataloader:
        if gt == 0:
            continue
        color_feats += get_feat(img_data, method='color')

    color_feats = color_feats / np.linalg.norm(color_feats)
    return color_feats
