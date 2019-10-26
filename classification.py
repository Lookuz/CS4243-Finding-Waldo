"""
    The core part: get the classifier
    TODO: adjust parameters provided
    TODO: implement KNN if necessary
"""

from scipy.spatial.distance import cdist
from sklearn import metrics as sk_metrics
from sklearn.svm import SVC

from dataset import *
from template import *


def fetch_feat_dict(class_name, method, overwrite=False):
    feat_dict_folder = os.path.join(os.getcwd(), f'{method}_dict')
    load_dir(feat_dict_folder)
    feat_dict_path = os.path.join(feat_dict_folder, f'{class_name}.npy')
    print('--- fetching the vocbulary')
    print(f'--- vocabulary stored in {feat_dict_path}')
    if overwrite or not os.path.exists(feat_dict_path):
        dataloader, _, _, _ = prepare_dataloader(class_name)
        if method is 'color':
            feat_dict = extract_color_features(dataloader)
        elif method is 'sift':
            feat_dict = extract_vocabs(dataloader, method=method,
                                       step=vocab_step, size=vocab_step)
        else:
            feat_dict = extract_vocabs(dataloader, method=method)
        np.save(feat_dict_path, feat_dict)
    return np.load(feat_dict_path)


def load_feats(loader, vocabs, method='sift', metrics='euclidean', **kwargs):
    vocab_num = len(vocabs)
    feats = []
    for img_data in loader:
        descriptors = get_feat(img_data, method=method, **kwargs)
        dists = cdist(descriptors, vocabs, metrics)
        classifications = np.argmin(dists, axis=1)
        occurences = np.bincount(classifications, minlength=vocab_num)
        hist_feature = occurences / np.linalg.norm(occurences)
        feats.append(hist_feature)
    return np.array(feats, dtype='f')


def trainer(class_name, method='sift', **kwargs):
    vocab_overwrite = kwargs.get('vocab_overwrite', False)
    tuning = kwargs.get('tuning', False)
    metrics = kwargs.get('metrics', 'euclidean')
    gamma = kwargs.get('gamma', 'scale')

    vocabs = fetch_feat_dict(class_name, method=method, overwrite=vocab_overwrite)
    pos_train_loader, pos_val_loader, neg_train_loader, neg_val_loader = prepare_dataloader(class_name, tuning=tuning)

    print('--- extracting features from the positive training set')
    pos_train_feats = load_feats(pos_train_loader, vocabs, method=method, metrics=metrics)
    print('--- extracting features from the negative training set')
    neg_train_feats = load_feats(neg_train_loader, vocabs, method=method, metrics=metrics)
    train_feats = np.vstack((pos_train_feats, neg_train_feats))
    pos_train_labels = np.ones(len(pos_train_feats))
    neg_train_labels = np.zeros(len(neg_train_feats))
    train_labels = np.hstack((pos_train_labels, neg_train_labels))

    print('--- SVM starts working')
    clf = SVC(gamma=gamma)
    clf.fit(train_feats, train_labels)

    print('--- extracting features from the positive validation set')
    pos_val_feats = load_feats(pos_val_loader, vocabs, method=method, metrics=metrics)
    print('--- extracting features from the negative validation set')
    neg_val_feats = load_feats(neg_val_loader, vocabs, method=method, metrics=metrics)
    val_feats = np.vstack((pos_val_feats, neg_val_feats))
    pos_val_labels = np.ones(len(pos_val_feats))
    neg_val_labels = np.zeros(len(neg_val_feats))
    val_labels = np.hstack((pos_val_labels, neg_val_labels))

    print('--- predicting the validation set labels')
    predict_result = clf.predict(val_feats)
    precision = sk_metrics.precision_score(y_true=val_labels, y_pred=predict_result)
    recall = sk_metrics.recall_score(y_true=val_labels, y_pred=predict_result)
    f1_score = sk_metrics.f1_score(y_true=val_labels, y_pred=predict_result)
    print('--- precision: %.3f, recall: %.3f, f1-score: %.3f' % (precision, recall, f1_score))

    return clf


