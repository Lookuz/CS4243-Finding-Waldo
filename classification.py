"""
    The core part: get the classifier
    TODO: adjust parameters provided
    TODO: implement KNN if necessary
"""

import pickle
import joblib

from scipy.spatial.distance import cdist
from sklearn import metrics as sk_metrics
from sklearn.svm import SVC

from dataset import *
from template import *

descriptor_size = {
    'color': 256,
    'sift': 128,
    'surf': 128,
    'akaze': 61
}

"""
    params:
    method: 'sift', 'surf', 'akaze'
    pos_classes: positive classes labels (ref: prepare_classification_dataloader)
    neg_classes: negative classes labels (ref: prepare_classification_dataloader)
    
"""
class Classifier:
    def __init__(self, method, pos_classes, neg_classes=None,
                 simple=True, neg_ratio=1.0, valid_ratio=0.2, **kwargs):
        self.method = method
        self.feat_extractor_attrs(**kwargs)

        self.pos_classes = list(pos_classes) if pos_classes else None
        self.neg_classes = list(neg_classes) if neg_classes else None

        train_loader, valid_loader = prepare_classification_dataloader(
            pos_classes=pos_classes, neg_classes=neg_classes,
            simple=simple, neg_ratio=neg_ratio, valid_ratio=valid_ratio)
        self.train_instances = list(train_loader)
        self.valid_instances = list(valid_loader)

        self.vocabs = None
        self.trained = False

    def feat_extractor_attrs(self, **kwargs):
        self.DSP_OBJ = None
        self.metrics = kwargs.get('metrics', 'euclidean')
        if self.method is 'surf':
            self.DSP_OBJ = kwargs.get('DSP_OBJ')
            if self.DSP_OBJ is None:
                surf_obj = cv.xfeatures2d.SURF_create(500)
                surf_obj.setUpright(True)
                surf_obj.setExtended(True)
                self.DSP_OBJ = surf_obj
        if self.method is 'akaze':
            self.DSP_OBJ = kwargs.get('DSP_OBJ')
            if self.DSP_OBJ is None:
                akaze_obj = cv.AKAZE_create()
                self.DSP_OBJ = akaze_obj


    def fetch_feat_dict(self):
        print(f'--- fetching the vocbulary for {self.method}')
        if self.method is 'color':
            self.vocabs = extract_color_features(self.train_instances)
        else:
            self.vocabs = extract_vocabs(
                self.train_instances, method=self.method, step=vocab_step,
                size=vocab_size, DSP_OBJ=self.DSP_OBJ)

    def load_feats(self, X):
        vocab_num = len(self.vocabs)
        feats = []
        for img_data in X:
            descriptors = get_feat(img_data, method=self.method,
                                   step=sample_step, size=sample_size,
                                   DSP_OBJ=self.DSP_OBJ)
            if descriptors is None:
                print('one image get empty descriptors for training or testing')
                descriptors = np.zeros((1, descriptor_size[self.method]))
            dists = cdist(descriptors, self.vocabs, self.metrics)
            classifications = np.argmin(dists, axis=1)
            occurences = np.bincount(classifications, minlength=vocab_num)
            hist_feature = occurences / np.linalg.norm(occurences)
            feats.append(hist_feature)
        return np.array(feats, dtype='f')

    def train(self, **kwargs):
        if self.vocabs is None:
            self.fetch_feat_dict()
        gamma = kwargs.get('gamma', 'scale')
        kernel = kwargs.get('kernel', 'rbf')

        print('--- extracting features from the training set')
        train_images, train_labels = zip(*self.train_instances)
        train_feats = self.load_feats(train_images)

        print('--- extracting features from the positive validation set')
        valid_images, valid_labels = zip(*self.valid_instances)
        valid_feats = self.load_feats(valid_images)

        self.clf = SVC(gamma=gamma, kernel=kernel, probability=True)
        self.clf.fit(train_feats, train_labels)

        print('--- predicting the validation set labels')
        predict_result = self.clf.predict(valid_feats)
        precision = sk_metrics.precision_score(y_true=valid_labels, y_pred=predict_result)
        recall = sk_metrics.recall_score(y_true=valid_labels, y_pred=predict_result)
        f1_score = sk_metrics.f1_score(y_true=valid_labels, y_pred=predict_result)
        print('--- precision: %.3f, recall: %.3f, f1-score: %.3f' % (precision, recall, f1_score))

        self.trained = True

    def predict(self, X):
        if not self.trained:
            print(f'use default paramters to train')
        feats = self.load_feats(X)
        return self.clf.predict(feats)


    def predict_proba(self, X):
        if not self.trained:
            print(f'use default paramters to train')
        feats = self.load_feats(X)
        return self.clf.predict_proba(feats)


class PreparedClassifier:
    def __init__(self, class_name, mode):
        cur_dir = os.getcwd()
        vocab_name = f'{class_name}_{mode}.pkl'
        model_name = f'{class_name}_{mode}.joblib'
        model_path = os.path.join(cur_dir, 'complex_models', model_name)
        vocab_path = os.path.join(cur_dir, 'complex_vocabs', vocab_name)
        with open(model_path, 'rb') as fp:
            self.clf = joblib.load(fp)
        with open(vocab_path, 'rb') as fp:
            self.vocabs = pickle.load(fp)
        self.method = 'surf' if mode == 'simple' else 'sift'

        self.DSP_OBJ = None
        if self.method == 'surf':
            self.DSP_OBJ = cv.xfeatures2d.SURF_create(500)
            self.DSP_OBJ.setUpright(True)
            self.DSP_OBJ.setExtended(True)
        self.metrics = 'euclidean'


    def load_feats(self, X):
        vocab_num = len(self.vocabs)
        feats = []
        for img_data in X:
            descriptors = get_feat(img_data, method=self.method,
                                   step=sample_step, size=sample_size,
                                   DSP_OBJ=self.DSP_OBJ)
            hist_feature = np.zeros(vocab_num)
            if descriptors is not None:
                dists = cdist(descriptors, self.vocabs, self.metrics)
                classifications = np.argmin(dists, axis=1)
                occurences = np.bincount(classifications, minlength=vocab_num)
                hist_feature = occurences / np.linalg.norm(occurences)
            feats.append(hist_feature)
        return np.array(feats, dtype='f')

    def predict(self, X):
        feats = self.load_feats(X)
        return self.clf.predict(feats)

    def predict_proba(self, X):
        feats = self.load_feats(X)
        return self.clf.predict_proba(feats)


