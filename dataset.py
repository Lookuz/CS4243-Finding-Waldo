"""
    Data collection and preprocessing
"""

import os
import cv2
import pickle
import numpy as np
import xml.etree.ElementTree as ET
import json

figure_classes = {'waldo', 'wenda', 'wizard'}
extra_poses_classified = ['waldo']


def load_dir(dir):
    try:
        os.makedirs(dir)
    except:
        FileExistsError

def load_full_subdir(dir):
    subdir = os.listdir(dir)
    return list(map(lambda x: os.path.join(dir, x), subdir))

def load_image(path):
    im = cv2.imread(path)
    im = im.astype(np.float32) / 255
    return im


def img_to_show(im):
    return im[:, :, ::-1]


def extract_annots(validation=False):
    dataset_type = 'val' if validation else 'train'
    goal_dir = os.getcwd()
    annot_source_path = os.path.join(goal_dir, 'datasets', 'Annotations')
    file_idx_path = os.path.join(goal_dir, 'datasets', 'ImageSets', f'{dataset_type}.txt')
    annots_path = os.path.join(goal_dir, 'cache_anno', f'{dataset_type}_annots.pkl')
    with open(file_idx_path, 'r') as fp:
        annot_idxs = fp.readlines()

    annots_result = {}
    for line in annot_idxs:
        idx = line.strip()
        annot = {}
        source_path = os.path.join(annot_source_path, f'{idx}.xml')
        tree = ET.parse(source_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            box = obj.find('bndbox')
            annot[name] = [
                int(box.find('xmin').text),
                int(box.find('ymin').text),
                int(box.find('xmax').text),
                int(box.find('ymax').text)]
        annots_result[idx] = annot
    with open(annots_path, 'wb') as fp:
        pickle.dump(annots_result, fp)


def save_provided_patches(validation=False):
    dataset_type = 'val' if validation else 'train'
    goal_dir = os.getcwd()
    annots_path = os.path.join(goal_dir, 'cache_anno', f'{dataset_type}_annots.pkl')
    images_src_path = os.path.join(goal_dir, 'datasets', 'JPEGImages')
    gt_des_path = os.path.join(goal_dir, 'datasets', dataset_type, 'positives')
    load_dir(gt_des_path)
    des_map = {
        'waldo': os.path.join(gt_des_path, 'waldo'),
        'wenda': os.path.join(gt_des_path, 'wenda'),
        'wizard': os.path.join(gt_des_path, 'wizard')
    }
    file_idx_path = os.path.join(goal_dir, 'datasets', 'ImageSets', f'{dataset_type}.txt')

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
            patch = img_data[box[1]:box[3]+1, box[0]:box[2]+1]
            save_path = os.path.join(des_map[name], f'gt_{gt_idx}.npy')
            np.save(save_path, patch)
            gt_idx += 1


def save_poses():
    # extract waldo from different images in image 1
    goal_dir = os.getcwd()
    images_src_path = os.path.join(goal_dir, 'datasets', 'JPEGImages')

    for class_name in extra_poses_classified:
        annots_path = os.path.join(goal_dir, 'datasets', 'Annotations', f'{class_name}_poses.json')
        gt_des_path = os.path.join(goal_dir, 'datasets', f'{class_name}_extra')
        load_dir(gt_des_path)
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


def prepare_dataset():
    save_provided_patches(True)
    save_provided_patches(False)
    save_poses()
    save_extra_patches()


def positive_source(name):
    goal_dir = os.getcwd()
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


def positive_loader(srcs):
    for img_path in srcs:
        yield np.load(img_path)


def negative_loader(srcs, rand_num=3):
    for img_path in srcs:
        yield np.load(img_path)
    for img_patch in load_random_patch(rand_num):
        yield img_patch


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
