"""
Data collection and preprocessing module
Author: Yang Si Han
Contributors: Lukaz
"""
import os
import cv2
import pickle
import numpy as np
import xml.etree.ElementTree as ET
import json
import shutil

figure_classes = {'waldo', 'wenda', 'wizard'}
extra_poses_classified = ['waldo']

""""""" Image Loading and Preprocessing """""""
def load_image(path):
    im = cv2.imread(path)
    im = im.astype(np.float32) / 255.
    return im

def img_to_show(im):
    return im[:, :, ::-1]

""""""""""""""""""""""""""""""""""""""""""""""""

""""""" Image File Path Extraction """""""
# Function that creates the directory specified
def load_dir(dir):
    try:
        os.makedirs(dir)
    except:
        FileExistsError

# Function that gets a list of file paths in the specified root directory
def load_full_subdir(dir):
    subdir = os.listdir(dir)
    return list(map(lambda x: os.path.join(dir, x), subdir))

# Function that removes specified directory
# Directory specified is recommended to be full path
def del_dir(dir):
    try:
        shutil.rmtree(dir)
    except OSError: # Directory invalid
        print('Invalid directory specified')

# Function that deletes files within the directory,
# But keeps the empty directory
def clean_dir(dir):
    try:
        for item in os.listdir(dir):
            file_path = os.path.join(dir, item)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)
    except OSError: # Directory invalid
        print('Invalid directory specified')

""""""""""""""""""""""""""""""""""""""""""""""""

""""""" Dataset Loading and Preprocessing """""""
# Function that extracts annotations for dataset
# If validation=True, then the annotations for validation set is extracted
# Else, the annotations for the train set is extracted instead
# Saves the extracted annotations in the form of a dictionary mapping
# Each image index to it's annotations
# Annotations: (May be updated)
# Bounding Box Coordinates
def extract_annots(validation=False):
    dataset_type = 'val' if validation else 'train'

    # Extract annotations file paths
    goal_dir = os.getcwd()
    annot_source_path = os.path.join(goal_dir, 'datasets', 'Annotations') # Default annotations
    file_idx_path = os.path.join(goal_dir, 'datasets', 'ImageSets', f'{dataset_type}.txt')
    annots_path = os.path.join(goal_dir, 'cache_anno', f'{dataset_type}_annots.pkl')
    with open(file_idx_path, 'r') as fp:
        annot_idxs = fp.readlines()

    annots_result = {} # Stores (image index : annotations)
    # Iterate through each index in the train/validation set
    for line in annot_idxs:
        idx = line.strip()
        annot = {}

        # Load current indexed dataset
        source_path = os.path.join(annot_source_path, f'{idx}.xml')
        tree = ET.parse(source_path)
        root = tree.getroot()

        # Parse parameters in XML annotation file
        for obj in root.findall('object'):
            name = obj.find('name').text
            # Extract bounding box coordinates
            box = obj.find('bndbox')
            annot[name] = [
                int(box.find('xmin').text),
                int(box.find('ymin').text),
                int(box.find('xmax').text),
                int(box.find('ymax').text)]
        annots_result[idx] = annot
    
    # Save the current annotations as pickled file
    with open(annots_path, 'wb') as fp:
        pickle.dump(annots_result, fp)

# Function that extracts positive feature patches from the dataset
# The positive feature patches are bounding boxes of images containing Waldo
# To be used in feature extraction
def save_provided_patches(validation=False):
    dataset_type = 'val' if validation else 'train'

    # Load file paths
    goal_dir = os.getcwd()
    annots_path = os.path.join(goal_dir, 'cache_anno', f'{dataset_type}_annots.pkl')
    images_src_path = os.path.join(goal_dir, 'datasets', 'JPEGImages')
    gt_des_path = os.path.join(goal_dir, 'datasets', dataset_type, 'positives')
    load_dir(gt_des_path) # Create new path to store extracted feature patches

    des_map = {
        'waldo': os.path.join(gt_des_path, 'waldo'),
        'wenda': os.path.join(gt_des_path, 'wenda'),
        'wizard': os.path.join(gt_des_path, 'wizard')
    }
    file_idx_path = os.path.join(goal_dir, 'datasets', 'ImageSets', f'{dataset_type}.txt')

    # Load saved annotations
    if not os.path.exists(annots_path):
        extract_annots(validation)
    with open(annots_path, 'rb') as fp:
        annots = pickle.load(fp)
    with open(file_idx_path, 'r') as fp:
        img_idxs = fp.readlines()
    for path in des_map.values():
        load_dir(path)

    gt_idx = 0
    for line in img_idxs:
        idx = line.strip()
        img_path = os.path.join(images_src_path, f'{idx}.jpg')
        img_data = cv2.imread(img_path)
        annot = annots[idx]
        for name, box in annot.items():
            # Extract current bounding box as patch
            patch = img_data[box[1]:box[3]+1, box[0]:box[2]+1]
            save_path = os.path.join(des_map[name], f'gt_{gt_idx}.npy')
            np.save(save_path, patch)
            gt_idx += 1

# NOTE: Image 1 in the dataset contains various poses and angles of Waldo
# Function extracts various poses of Waldo from this image
def save_poses():
    # Extract Waldo from different images in image 1
    goal_dir = os.getcwd()
    images_src_path = os.path.join(goal_dir, 'datasets', 'JPEGImages')

    for class_name in extra_poses_classified:
        # Load file path for saving extra poses
        annots_path = os.path.join(goal_dir, 'datasets', 'Annotations', f'{class_name}_poses.json')
        gt_des_path = os.path.join(goal_dir, 'datasets', f'{class_name}_extra')
        load_dir(gt_des_path)

        # TODO: Is extract annots supposed to produce json file?
        if not os.path.isfile(annots_path):
            extract_annots()
        with open(annots_path) as fp:
            pose_annots = json.load(fp)

        des_map = {
            'front': os.path.join(gt_des_path, 'front'),
            'side': os.path.join(gt_des_path, 'side'),
            'tilt': os.path.join(gt_des_path, 'tilt')
        }

        pos_idx = 0
        meta_data = pose_annots['_via_img_metadata']
        # Create file path for each pose
        for path in des_map.values():
            load_dir(path)
        
        # Save different poses for Waldo
        for img_src in meta_data.values():
            bndboxes = img_src['regions']
            img_data = cv2.imread(os.path.join(images_src_path, img_src['filename']))
            for box in bndboxes:
                box_rect_info = box['shape_attributes']
                x_min = box_rect_info['x']
                y_min = box_rect_info['y']
                x_max = x_min + box_rect_info['width']
                y_max = y_min + box_rect_info['height']
                patch_data = img_data[y_min:y_max, x_min:x_max]
                label = box['region_attributes']['pose']
                save_path = os.path.join(des_map[label], f'{class_name}_{pos_idx}.npy')
                np.save(save_path, patch_data)
                pos_idx += 1


def save_extra_patches():
    goal_dir = os.getcwd()
    annots_path = os.path.join(goal_dir, 'datasets', 'Annotations', 'headshot_candidates.json')
    images_src_path = os.path.join(goal_dir, 'datasets', 'JPEGImages')
    gt_des_path = os.path.join(goal_dir, 'datasets', 'extra', 'positives')
    cf_des_path = os.path.join(goal_dir, 'datasets', 'extra', 'confusion')
    load_dir(gt_des_path)
    load_dir(cf_des_path)
    with open(annots_path) as fp:
        examples_annots = json.load(fp)
    meta_data = examples_annots['_via_img_metadata']

    labels = ['waldo', 'wenda', 'wizard', 'close', 'close_wizard', 'face', 'neg']
    des_map = {
        'waldo': os.path.join(gt_des_path, 'waldo'),
        'wenda': os.path.join(gt_des_path, 'wenda'),
        'wizard': os.path.join(gt_des_path, 'wizard'),
        'close': os.path.join(cf_des_path, 'waldo_wenda'),
        'close_wizard': os.path.join(cf_des_path, 'wizard'),
        'face': os.path.join(cf_des_path, 'waldo_wenda'),
        'neg': os.path.join(cf_des_path, 'waldo_wenda'),
    }

    example_id = 0
    for path in des_map.values():
        load_dir(path)
    for img_src in meta_data.values():
        bndboxes = img_src['regions']
        img_data = cv2.imread(os.path.join(images_src_path, img_src['filename']))
        for box in bndboxes:
            box_rect_info = box['shape_attributes']
            x_min = box_rect_info['x']
            y_min = box_rect_info['y']
            x_max = x_min + box_rect_info['width']
            y_max = y_min + box_rect_info['height']
            patch_data = img_data[y_min:y_max, x_min:x_max]
            label = box['region_attributes']['classification']
            if label in labels:
                save_path = os.path.join(des_map[label], f'extra_{example_id}.npy')
                np.save(save_path, patch_data)
                example_id += 1

# Returns the file paths for the positive data for both training and validation sets
# Returns the datasets in the form of tuple of lists
def positive_source(name):
    goal_dir = os.getcwd()
    # Loads dataset depending on the name of object to be detected
    train_img_src = os.path.join(goal_dir, 'datasets', 'train', 'positives', name)
    val_img_src = os.path.join(goal_dir, 'datasets', 'val', 'positives', name)
    extra_img_src = os.path.join(goal_dir, 'datasets', 'extra', 'positives', name)

    train_srcs = load_full_subdir(train_img_src)
    val_srcs = load_full_subdir(val_img_src)
    extra_srcs = load_full_subdir(extra_img_src)

    extra_pose_path = os.path.join(goal_dir, 'datasets', f'{name}_extra')
    if os.path.exists(extra_pose_path):
        srcs = []
        for dir in load_full_subdir(extra_pose_path):
            srcs.extend(load_full_subdir(dir))
        extra_srcs.extend(srcs)

    return train_srcs, val_srcs, extra_srcs

# Returns the file paths for the negative data for both training and validation sets
# Returns the datasets in the form of tuple of lists
def negative_source(name, tuning=False):
    train_srcs = []
    val_srcs = []
    extra_srcs = []

    for class_name in figure_classes - {name}:
        t, v, e = positive_source(class_name)
        train_srcs.extend(t)
        val_srcs.extend(v)
        extra_srcs.extend(e)

    if tuning:
        goal_dir = os.getcwd()
        confusion_name = 'wizard' if name is 'wizard' else 'waldo_wenda'
        confusion_path = os.path.join(goal_dir, 'datasets', 'extra', 'confusion', confusion_name)
        extra_srcs.extend(load_full_subdir(confusion_path))

    return train_srcs, val_srcs, extra_srcs


# Loads image sets with positive labels from given dataset source paths
# Source paths can be loaded from the positive_source function
def positive_loader(srcs):
    for img_path in srcs:
        yield np.load(img_path)

# Loads image sets with negative labels from given dataset source paths
# Source paths can be loaded from the negative_source function
def negative_loader(srcs, rand_num=3):
    for img_path in srcs:
        yield np.load(img_path)
    for img_patch in load_random_patch(rand_num):
        yield img_patch

# Function that extracts and preprocesses the given dataset
# If clean=True is specified, then existing files will be deleted
# before the preprocessed files are added again. This is to allow for a
# fresh re-loading of data
def prepare_dataset(clean=False):
    curr_wd = os.getcwd()

    # Extract Annotations
    annot_source_path = os.path.join(curr_wd, 'cache_anno')
    assert os.path.exist(annot_source_path)
    if clean:
        clean_dir(annot_source_path)
    annot_source_path_train = os.path.join(annot_source_path, 'train_annots.pkl')
    annot_source_path_val = os.path.join(annot_source_path, 'val_annots.pkl')
    if not os.path.exists(annot_source_path_train):
        extract_annots(validation=False)
    if not os.path.exists(annot_source_path_val):
        extract_annots(validation=True)

    # Extract Window Patches
    patch_source_path = os.path.join(curr_wd, 'datasets')
    assert os.path.exist(patch_source_path)
    patch_source_path_train = os.path.join(patch_source_path, 'train')
    patch_source_path_val = os.path.join(patch_source_path, 'val')
    if clean:
        del_dir(patch_source_path_train)
        del_dir(patch_source_path_val)
    if not os.path.exists(patch_source_path_train):
        save_provided_patches(validation=False)
    if not os.path.exists(patch_source_path_val):
        save_provided_patches(validation=True)

    pose_source_path = os.path.join(curr_wd, 'datasets', 'waldo_extra')
    if clean:
        del_dir(waldo_extra)
    if not os.path.exists(pose_source_path):
        save_poses()

    extra_poses_positive_path = os.path.join(curr_wd, 'datasets', 'extra', 'positives')
    extra_poses_confusion_path = os.path.join(curr_wd, 'datasets', 'extra', 'confusion')
    if clean:
        del_dir(extra_poses_positive_path)
        del_dir(extra_poses_confusion_path)
    if not os.path.exists(extra_poses_confusion_path) or not os.path.exists(extra_poses_positive_path):
        save_extra_patches()

""""""""""""""""""""""""""""""""""""""""""""""""

def load_random_patch(num_per_img):
    # image 001, 004, 006, 032, 043, 048, 068 not used
    goal_dir = os.getcwd()
    images_src_path = os.path.join(goal_dir, 'datasets', 'JPEGImages')
    train_file_idx_path = os.path.join(goal_dir, 'datasets', 'ImageSets', 'good_train.txt')
    with open(train_file_idx_path, 'r') as fp:
        img_idxs = fp.readlines()
    for line in img_idxs:
        idx = line.strip()
        img_path = os.path.join(images_src_path, f'{idx}.jpg')
        img_data = cv2.imread(img_path)
        im_h, im_w, _ = img_data.shape
        x = np.random.randint(0, im_w - 100, )
        y = np.random.randint(0, im_h - 200)
        widths = np.random.randint(40, 100, num_per_img)
        ratio = 1 + np.random.random(num_per_img)
        heights= np.multiply(widths, ratio).astype(int)
        for idx in range(num_per_img):
            patch = img_data[y:y+heights[idx],
                             x:x+widths[idx]]
            yield patch

"""
For use with classification.py
Author: Yang Si Han
"""
# Prepares the training and validation dataset by loading 
# The images from the positive and negative datasets
def prepare_dataloader(class_name, tuning=False, val_propotion=0.2):
    # return positive training loader
    # return positive validation loader
    # return negative training loader
    # return negative validation loader
    pos_train, pos_val, pos_extra = positive_source(class_name)
    neg_train, neg_val, neg_extra = negative_source(class_name, tuning=tuning)

    np.random.shuffle(pos_extra)
    np.random.shuffle(neg_extra)
    num_extra_pos_val = int(len(pos_extra)*val_propotion)
    num_extra_neg_val = int(len(neg_extra)*val_propotion)
    pos_val.extend(pos_extra[:num_extra_pos_val])
    pos_train.extend(pos_extra[num_extra_pos_val:])
    neg_val.extend(neg_extra[:num_extra_neg_val])
    neg_train.extend(neg_extra[num_extra_neg_val:])

    return positive_loader(pos_train), positive_loader(pos_val), \
           negative_loader(neg_train), negative_loader(neg_val)
