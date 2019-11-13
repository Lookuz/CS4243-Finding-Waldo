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
import random

figure_classes = {'waldo', 'wenda', 'wizard'}

model_file_path = 'datasets/models/' # Default file path for saving classifiers

""""""" Image Loading and Preprocessing """""""
def load_image(path):
    im = cv2.imread(path)
    im = im.astype(np.float32) / 255.
    return im

def img_to_show(im):
    return im[:, :, ::-1]

def to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
    result = []
    subdirs = os.listdir(dir)
    for pth in subdirs:
        full_path = os.path.join(dir, pth)
        if os.path.isdir(full_path):
            result += load_full_subdir(full_path)
        else:
            result.append(full_path)
    return result


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
def extract_provided_annots():
    goal_dir = os.getcwd()
    annot_source_path = os.path.join(goal_dir, 'datasets', 'Annotations')

    # clean previous result
    annots_desc_path = os.path.join(goal_dir, 'cache_anno', 'classification')
    load_dir(annots_desc_path)
    annots_path = os.path.join(annots_desc_path, 'provided_annots.pkl')
    if os.path.exists(annots_path):
        os.remove(annots_path)

    annots_result = {} # Stores (image index : [coord, label])
    xml_paths = os.listdir(annot_source_path)
    for pth in xml_paths:
        img_name, suffix = pth.rsplit('.', 1)
        if suffix != 'xml':
            continue  # only process xml files

        img_name = f'{img_name}.jpg'
        pth = os.path.join(annot_source_path, pth)
        annots_result[img_name] = []
        tree = ET.parse(pth)
        root = tree.getroot()

        # Parse parameters in XML annotation file
        for obj in root.findall('object'):
            label = [obj.find('name').text]
            # Extract bounding box coordinates
            box = obj.find('bndbox')
            coord = [
                int(box.find('xmin').text),
                int(box.find('ymin').text),
                int(box.find('xmax').text),
                int(box.find('ymax').text)]
            annots_result[img_name].append([
                coord, label])
    
    # Save the current annotations as pickled file
    with open(annots_path, 'wb') as fp:
        pickle.dump(annots_result, fp)


"""
function to extract annotation from JSON
"""
def extract_extra_annots():
    goal_dir = os.getcwd()
    annots_path = os.path.join(goal_dir, 'datasets', 'Annotations')
    face_annot_pth = os.path.join(annots_path, 'headshots.json')
    body_annot_pth = os.path.join(annots_path, 'full_body.json')
    waldo_pose_annot_pth = os.path.join(annots_path, 'waldo_poses.json')

    annots_desc_path = os.path.join(goal_dir, 'cache_anno', 'classification')
    load_dir(annots_desc_path)
    annots_desc_path = os.path.join(annots_desc_path, 'extra_annots.pkl')
    if os.path.exists(annots_desc_path):
        os.remove(annots_desc_path)

    """
    the structure of data:
    waldo:
    |__ face
    |   |__front
    |   |__side
    |__ body
        |__half
        |__full
    wenda:
    wizard
    """
    with open(face_annot_pth) as fp:
        face_annots = json.load(fp)
    with open(body_annot_pth) as fp:
        body_annots = json.load(fp)
    with open(waldo_pose_annot_pth) as fp:
        waldo_pose_annots = json.load(fp)

    meta_data = {
        'face': face_annots['_via_img_metadata'],
        'body': body_annots['_via_img_metadata'],
        'pose': waldo_pose_annots['_via_img_metadata'],
    }

    label_class = {
        'face': 'classification',
        'body': 'Body',
        'pose': 'pose',
    }

    label_mapping = {
        'waldo_body': ['waldo', 'body', 'full'],
        'waldo_half_body': ['waldo', 'body', 'half'],
        'wenda_body': ['wenda', 'body', 'full'],
        'wenda_half_body': ['wenda', 'body', 'half'],
        'wizard_body': ['wizard', 'body', 'full'],
        'wizard_half_body': ['wizard', 'body', 'half'],
        'false_body': ['other', 'body', 'full'],
        'false_half_body': ['other', 'body', 'half'],
        'waldo': ['waldo', 'face', 'front'],
        'wenda': ['wenda', 'face', 'front'],
        'wizard': ['wizard', 'face', 'front'],
        'neg': ['other', 'face', 'front'],
        'front': ['waldo', 'face', 'front'],
        'tilt': ['waldo', 'face', 'front'],
        'side': ['waldo', 'face', 'side'],
    }

    annots = {}

    # classify the bounding boxes
    # annotations are stored as
    # [[x_min, y_min, x_max, y_max], [label]]
    for k,v in meta_data.items():
        label_type = label_class[k]
        for img in v.values():
            img_name = img['filename']
            if img_name not in annots:
                annots[img_name] = []

            for region in img['regions']:
                region_label = region['region_attributes'][label_type]
                rect_info = region['shape_attributes']
                x_min = rect_info['x']
                y_min = rect_info['y']
                x_max = x_min + rect_info['width']
                y_max = y_min + rect_info['height']
                coord = [x_min, y_min, x_max, y_max]
                label = list(label_mapping[region_label])
                annots[img_name].append([coord, label])

    with open(annots_desc_path, 'wb') as fp:
        pickle.dump(annots, fp)


"""
images preprocessing
"""
def preprocess_img(img):
    result = []
    result.append(img)
    gaussian_smooth = cv2.GaussianBlur(img,(5,5),0)
    result.append(gaussian_smooth)
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist_smooth = cv2.equalizeHist(grey_img)
    result.append(hist_smooth)
    return result


"""
patches preprocessing and augmentation
"""
def preprocess_patch(imgs, xmin, ymin, xmax, ymax):
    patches = []
    for img in imgs:
        patch = img[ymin:ymax, xmin:xmax]
        patches.append(patch)
        flip_patch = cv2.flip(patch, 1)
        patches.append(flip_patch)
    return patches


"""
load extra annotated patches
if image list is provided (i.e.: ['001', '005', ...])
only patches in the list will be saved
"""
def save_patches(img_lst=None, is_provided=True):
    source_type = 'provided' if is_provided else 'extra'

    goal_dir = os.getcwd()
    annots_path = os.path.join(goal_dir, 'cache_anno', 'classification', f'{source_type}_annots.pkl')
    images_src_path = os.path.join(goal_dir, 'datasets', 'JPEGImages')
    images_dsc_path = os.path.join(goal_dir, 'datasets', 'classification', source_type)

    if os.path.exists(images_dsc_path):
        clean_dir(images_dsc_path)
    load_dir(images_dsc_path)

    with open(annots_path, 'rb') as fp:
        annots = pickle.load(fp)

    for img_file_name, boxes in annots.items():
        img_name = img_file_name.rsplit('.', 1)[0]
        if img_lst and img_name not in img_lst:
            continue

        img_src_path = os.path.join(images_src_path, img_file_name)
        img_data = cv2.imread(img_src_path)
        imgs = preprocess_img(img_data)

        for idx, box in enumerate(boxes):
            coord, label = box
            patches = preprocess_patch(imgs, *coord)

            dsc_path = os.path.join(images_dsc_path, *label)
            load_dir(dsc_path)
            for i, patch in enumerate(patches):
                result_file_name = f'{img_name}_{idx}_{i}.npy'
                result_path = os.path.join(dsc_path, result_file_name)
                np.save(result_path, patch)


"""
load random patches from list images
img_lst: ['001', '005', ...]
"""
def save_random_patches(img_lst, num_per_img):
    # image 001, 004, 006, 032, 043, 048, 068 should not be used
    bad_sources = ['001', '004', '006', '032', '043', '048', '068']

    goal_dir = os.getcwd()
    images_src_path = os.path.join(goal_dir, 'datasets', 'JPEGImages')
    images_dsc_path = os.path.join(goal_dir, 'datasets', 'classification', 'random')

    load_dir(images_dsc_path)
    clean_dir(images_dsc_path)

    scale_factor = {
        'face': (1.0, 1.0),
        'half': (1.5, 1.0),
        'full': (2.0, 1.5),
    }

    for img_name in img_lst:
        # DONNOT load from these images
        if img_name in bad_sources:
            continue

        img_file_name = f'{img_name}.jpg'
        img_src_path = os.path.join(images_src_path, img_file_name)
        img_data = cv2.imread(img_src_path)
        imgs = preprocess_img(img_data)
        im_h, im_w, _ = img_data.shape

        for k, v in scale_factor.items():
            folder_pth = os.path.join(images_dsc_path, k)
            load_dir(folder_pth)
            x = np.random.randint(0, im_w - 150, num_per_img)
            y = np.random.randint(0, im_h - 300, num_per_img)
            widths = np.random.randint(40, 150, num_per_img)
            ratio = v[0] + np.random.random(num_per_img) * v[1]
            heights = np.multiply(widths, ratio).astype(int)

            for idx in range(num_per_img):
                patches = preprocess_patch(imgs, x[idx], y[idx],
                                           x[idx]+widths[idx], y[idx]+heights[idx])
                for i, patch in enumerate(patches):
                    result_file_name = f'{img_name}_{idx}_{i}.npy'
                    result_path = os.path.join(folder_pth, result_file_name)
                    np.save(result_path, patch)


def save_random_images(img_lst, num_per_img):
    # image 001, 004, 006, 032, 043, 048, 068 should not be used
    bad_sources = ['001', '004', '006', '032', '043', '048', '068']

    goal_dir = os.getcwd()
    images_src_path = os.path.join(goal_dir, 'datasets', 'JPEGImages')
    images_dsc_path = os.path.join(goal_dir, 'datasets', 'bg')

    load_dir(images_dsc_path)
    clean_dir(images_dsc_path)

    scale_factor = {
        'face': (1.0, 1.0),
        'half': (1.5, 1.0),
        'full': (2.0, 1.5),
    }

    total_idx = 0
    for img_name in img_lst:
        # DONNOT load from these images
        if img_name in bad_sources:
            continue

        img_file_name = f'{img_name}.jpg'
        img_src_path = os.path.join(images_src_path, img_file_name)
        img_data = cv2.imread(img_src_path)
        im_h, im_w, _ = img_data.shape

        for k, v in scale_factor.items():
            x = np.random.randint(0, im_w - 150, num_per_img)
            y = np.random.randint(0, im_h - 300, num_per_img)
            widths = np.random.randint(40, 150, num_per_img)
            ratio = v[0] + np.random.random(num_per_img) * v[1]
            heights = np.multiply(widths, ratio).astype(int)
            x_end = x + widths
            y_end = y + heights

            for idx in range(num_per_img):
                patch = img_data[y[idx]:y_end[idx], x[idx]:x_end[idx]]
                result_file_name = f'{total_idx}.jpg'
                result_path = os.path.join(images_dsc_path, result_file_name)
                cv2.imwrite(result_path, patch)
                total_idx += 1


# Loads image from the provided paths list
def data_loader(instances):
    for instance in instances:
        img_path, label = instance
        yield np.load(img_path), label


"""
    Prepare the datasets
    
    Params:
    =======
    img_lst:
        only images in the the list will be loaded 
        into the dataset
    clean:
        whether to clean the previous dataset
    num_random:
        number of random patches generated
        from every image

"""
# Function that extracts and preprocesses the given dataset
# If clean=True is specified, then existing files will be deleted
# before the preprocessed files are added again. This is to allow for a
# fresh re-loading of data
def prepare_classification_dataset(img_lst, clean=True, num_random=20):
    curr_wd = os.getcwd()

    # Extract Annotations
    annot_source_path = os.path.join(curr_wd, 'cache_anno', 'classification')
    assert os.path.exists(annot_source_path)
    if clean:
        clean_dir(annot_source_path)
    provided_annot_source_path = os.path.join(annot_source_path, 'provided_annots.pkl')
    extra_annot_source_path = os.path.join(annot_source_path, 'extra_annots.pkl')
    if not os.path.exists(provided_annot_source_path):
        extract_provided_annots()
    if not os.path.exists(extra_annot_source_path):
        extract_extra_annots()

    # Extract provided Window Patches
    patch_source_path = os.path.join(curr_wd, 'datasets', 'classification')
    if clean:
        del_dir(patch_source_path)
    if not os.path.exists(patch_source_path):
        save_patches(img_lst, is_provided=True)
        save_patches(img_lst, is_provided=False)
        save_random_patches(img_lst=img_lst, num_per_img=num_random)


"""
    Prepares the training and validation dataset
    
    Parameters:
    ==========
    pos_classes:
        list of labels treated as positive examples
        each class name
        (   
            1 of ['waldo', 'wenda', 'wizard', 'other']
          + 1 of [['face'] + 1 of ['front', 'side']
                  ['body'] + 1 of ['half', 'full']]
        )
        example: 'waldo_face_front'
    simple:
        If the mode is set as simple, neg_classes will be ignored
        and random patches will be used as default negative.
        Otherwise, neg_classes will be used
    neg_ratio:
        control the number of negative examples.
        neg_ratio * 20 is the number of random patches be loaded
        if the mode is set to simple
    valid_ratio:
        ratio of data used for validation.

    Usage:
    =====
        prepare_classification_dataloader(
            pos_classes=['waldo_face_front', 'wenda_face_front'],
            neg_classes=['other_face_front'],
            simple=False, neg_ratio=0.7, valid_ratio=0.2)
            
    Return:
    ======
        return dataloaders for training and validation
"""
def prepare_classification_dataloader(pos_classes, neg_classes=None, simple=True,
                                      neg_ratio=1.0, valid_ratio=0.2):
    goal_dir = os.getcwd()
    instances = []
    random_types = []

    for class_name in pos_classes:
        sub_route = class_name.split('_')
        assert len(sub_route) == 3
        new_rnd_type = sub_route[1] if sub_route[1] == 'face' else sub_route[2]
        if new_rnd_type not in random_types:
            random_types.append(new_rnd_type)

        patch_folder_pth = os.path.join(goal_dir, 'datasets', 'classification', 'extra', *sub_route)
        patch_dirs = load_full_subdir(patch_folder_pth)
        for patch_dir in patch_dirs:
            instance = [patch_dir, 1]
            instances.append(instance)

    if simple or not neg_classes:
        neg_patch_folder_pths = [os.path.join(goal_dir, 'datasets', 'classification', 'random', rnd_type)
                                 for rnd_type in random_types]
    else:
        neg_patch_folder_pths = [os.path.join(goal_dir, 'datasets', 'classification', 'extra', *(cls_type.split('_')))
                                 for cls_type in neg_classes]

    for neg_patch_folder_pth in neg_patch_folder_pths:
        neg_instances = []
        neg_patch_dirs = load_full_subdir(neg_patch_folder_pth)
        for neg_patch_dir in neg_patch_dirs:
            instance = [neg_patch_dir, 0]
            neg_instances.append(instance)
        num_neg_instance = len(neg_instances)
        num_neg_instance = int(num_neg_instance * neg_ratio)
        random.shuffle(neg_instances)
        instances += neg_instances[:num_neg_instance]

    num_instance = len(instances)
    num_validation = int(num_instance * valid_ratio)
    random.shuffle(instances)

    print(f"---num of training instances: {num_instance - num_validation}")
    print(f"---num of validation instances: {num_validation}")

    return data_loader(instances[num_validation:]), \
           data_loader(instances[:num_validation])

# Function that extracts the training examples and labels
# from the provided training_instances
# Crafted to be used in conjunction with prepare_classification_loader
# which loads the training_instances
def extract_data(data):
    examples = [cv2.cvtColor(x[0].astype('uint8'), cv2.COLOR_BGR2RGB) if len(x[0].shape) == 3 else x[0] for x in data] # Extract examples
    labels = [x[1] for x in data] # Extract labels

    return examples, labels

# Function that retrieves the file paths for validation images
def get_val_image_paths():
    curr_wd = os.getcwd()

    # Get validation image numbers
    val_set_path = os.path.join(curr_wd, 'datasets/', 'ImageSets/', 'val.txt')
    val_images_index = []
    with open(val_set_path, 'r') as f:
        for line in f:
            val_images_index.append(line.strip())
            
    # Get validation image paths
    image_directory = os.path.join(curr_wd, 'datasets/', 'JPEGImages/')
    image_paths = load_full_subdir(image_directory)

    val_images_path = []
    for path in image_paths:
        filename, extension = os.path.splitext(os.path.basename(path))
        if extension == '.jpg' and filename in val_images_index:
            val_images_path.append(path)

    return val_images_path