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


def construct_training_data_per_class(class_name):
