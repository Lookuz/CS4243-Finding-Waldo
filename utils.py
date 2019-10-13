import os
import cv2
import pickle
import numpy as np
import xml.etree.ElementTree as ET
import json

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

def extract_train_annots():
    goal_dir = os.getcwd()
    annot_source_path = os.path.join(goal_dir, 'datasets', 'Annotations')
    train_file_idx_path = os.path.join(goal_dir, 'datasets', 'ImageSets', 'train.txt')
    train_annots_path = os.path.join(goal_dir, 'cache_anno', 'train_annots.pkl')
    with open(train_file_idx_path, 'r') as fp:
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
    with open(train_annots_path, 'wb') as fp:
        pickle.dump(annots_result, fp)


def save_templates():
    goal_dir = os.getcwd()
    annots_path = os.path.join(goal_dir, 'cache_anno', 'train_annots.pkl')
    images_src_path = os.path.join(goal_dir, 'datasets', 'JPEGImages')
    gt_des_path = os.path.join(goal_dir, 'datasets', 'positives')
    load_dir(gt_des_path)
    des_map = {
        'waldo': os.path.join(gt_des_path, 'waldo', 'unclassified'),
        'wenda': os.path.join(gt_des_path, 'wenda'),
        'wizard': os.path.join(gt_des_path, 'wizard')
    }
    train_file_idx_path = os.path.join(goal_dir, 'datasets', 'ImageSets', 'train.txt')

    with open(annots_path, 'rb') as fp:
        annots = pickle.load(fp)
    with open(train_file_idx_path, 'r') as fp:
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


def save_waldo_poses():
    # extract waldo from different images in image 1
    goal_dir = os.getcwd()
    annots_path = os.path.join(goal_dir, 'datasets', 'Annotations', 'waldo_poses.json')
    images_src_path = os.path.join(goal_dir, 'datasets', 'JPEGImages')
    gt_des_path = os.path.join(goal_dir, 'datasets', 'positives', 'waldo')
    load_dir(gt_des_path)
    with open(annots_path) as fp:
        waldo_pose_annots = json.load(fp)

    des_map = {
        'front': os.path.join(gt_des_path, 'front'),
        'side': os.path.join(gt_des_path, 'side'),
        'tilt': os.path.join(gt_des_path, 'tilt')
    }

    waldo_pos_idx = 0
    meta_data = waldo_pose_annots['_via_img_metadata']
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
            save_path = os.path.join(des_map[label], f'waldo_{waldo_pos_idx}.npy')
            np.save(save_path, patch_data)
            waldo_pos_idx += 1


def save_neg_patches():
    goal_dir = os.getcwd()
    annots_path = os.path.join(goal_dir, 'datasets', 'Annotations', 'headshot_candidates.json')
    images_src_path = os.path.join(goal_dir, 'datasets', 'JPEGImages')
    gt_des_path = os.path.join(goal_dir, 'datasets', 'positives')
    potential_des_path = os.path.join(goal_dir, 'datasets', 'potential')
    load_dir(gt_des_path)
    load_dir(potential_des_path)
    with open(annots_path) as fp:
        examples_annots = json.load(fp)
    meta_data = examples_annots['_via_img_metadata']

    labels = ['waldo', 'wenda', 'wizard', 'close', 'close_wizard', 'face', 'neg']
    des_map = {
        'waldo': os.path.join(gt_des_path, 'waldo', 'unclassified'),
        'wenda': os.path.join(gt_des_path, 'wenda'),
        'wizard': os.path.join(gt_des_path, 'wizard'),
        'close': os.path.join(potential_des_path, 'waldo_wenda'),
        'close_wizard': os.path.join(potential_des_path, 'wizard'),
        'face': os.path.join(potential_des_path, 'waldo_wenda'),
        'neg': os.path.join(potential_des_path, 'waldo_wenda'),
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
        im_h, im_w = img_data.shape
        x = np.random.randint(0, im_w - 100, )
        y = np.random.randint(0, im_h - 200)
        widths = np.random.randint(40, 100, num_per_img)
        ratio = 1 + np.random.random(num_per_img)
        heights= np.multiply(widths, ratio).astype(int)
        for idx in range(num_per_img):
            patch = img_data[y:y+heights(idx),
                             x:x+widths(idx)]
            yield patch


def classification_training_patches():
    # from gt
    save_templates()
    save_neg_patches()
    save_waldo_poses()
