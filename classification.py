from scipy.spatial.distance import cdist
from sklearn.svm import SVC

from utils import *
from template import *


def fetch_feat_dict(class_name, overwrite=True):
    feat_dict_path = os.path.join(os.getcwd(), 'feat_dict')
    pos_color_feat_path = os.path.join(feat_dict_path, 'pos_color', f'{class_name}.npy')
    pot_color_feat_path = os.path.join(feat_dict_path, 'pot_color', f'{class_name}.npy')
    pos_contour_feat_path = os.path.join(feat_dict_path, 'pos_contour', f'{class_name}.npy')
    pot_contour_feat_path = os.path.join(feat_dict_path, 'pot_contour', f'{class_name}.npy')
    if overwrite or \
        not os.path.exists(pos_color_feat_path) or \
        not os.path.exists(pot_color_feat_path):
        pos_color, pot_color = extract_color_features(class_name)
        np.save(pos_color_feat_path, pos_color)
        np.save(pot_color_feat_path, pos_color)
    if overwrite or \
        not os.path.exists(pos_contour_feat_path) or \
        not os.path.exists(pot_contour_feat_path):
        pos_contour, pot_contour = extract_vocabs(class_name)
        np.save(pos_color_feat_path, pos_contour)
        np.save(pot_color_feat_path, pos_contour)
    return {
        'pos_color': np.load(pos_color_feat_path),
        'pot_color': np.load(pot_contour_feat_path),
        'pos_contour': np.load(pos_contour_feat_path),
        'pot_contour': np.load(pot_contour_feat_path)
    }


def load_contour_feats(metric, vocabs, loader, *args):
    vocab_num = len(vocabs)
    feats = []
    for img_data in loader(*args):
        descriptors = get_contour_feat_sample(img_data)
        dists = cdist(descriptors, vocabs, metric)
        classifications = np.argmin(dists, axis=1)
        occurences = np.bincount(classifications, minlength=vocab_num)
        hist_feature = occurences / np.linalg.norm(occurences)
        feats.append(hist_feature)
    return np.array(feats, dtype='f')


def load_color_feats(loader, *args):
    feats = []
    for img_data in loader(*args):
        color_hist = get_color_feat(img_data)
        feats.append(color_hist)
    return np.array(feats, dtype='f')


def contour_classifier(class_name, sub_correct=False, metric='euclidean'):
    vocabs = extract_vocabs(class_name, sub_correct)
    pos_loader = load_sub_positive_patch if sub_correct else load_positive_patch
    neg_loader = get_contour_feat_sample

    pos_feats = load_contour_feats(metric, vocabs, pos_loader, class_name)
    neg_feats = load_contour_feats(metric, vocabs, neg_loader, 3)

    pos_labels = np.ones(len(pos_feats))
    neg_labels = np.zeros(len(neg_feats))

    X = np.vstack((pos_feats, neg_feats))
    Y = np.vstack((pos_labels, neg_labels))
    clf = SVC(gamma='scale')
    clf.fit(X, Y)
    return clf


def color_classifier(class_name, sub_correct=False):
    # color_pattern = extract_color_features(class_name, sub_correct)
    pos_loader = load_sub_positive_patch if sub_correct else load_positive_patch
    neg_loader = get_contour_feat_sample

    pos_feats = load_color_feats(pos_loader, class_name)
    neg_feats = load_color_feats(neg_loader, class_name)
    pos_labels = np.ones(len(pos_feats))
    neg_labels = np.zeros(len(neg_feats))

    X = np.vstack((pos_feats, neg_feats))
    Y = np.vstack((pos_labels, neg_labels))
    clf = SVC(gamma='scale')
    clf.fit(X, Y)
    return clf

