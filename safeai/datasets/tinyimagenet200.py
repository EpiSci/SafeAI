import os
import logging
import zipfile

import PIL
import numpy as np
import skimage
from tensorflow.keras.utils import get_file


def load_caches(cache_path):
    num_classes = 200
    num_samples_train = 500
    num_samples_test = 50

    train_y = np.array(np.arange(num_classes)               \
                .repeat(num_samples_train), dtype=np.uint8) \
                .reshape((-1, 1))                           \

    test_y = np.array(np.arange(num_classes)                \
               .repeat(num_samples_test), dtype=np.uint8)   \
               .reshape((-1, 1))                            \

    train_x = np.load(cache_path['train_x'])
    test_x = np.load(cache_path['test_x'])
    train_boxes = np.load(cache_path['train_boxes'])
    test_boxes = np.load(cache_path['test_boxes'])
    label_names = np.load(cache_path['label_names'])

    return (train_x, train_y), (test_x, test_y), \
           train_boxes, test_boxes, label_names


def save_caches(ret, cache_path):

    (train_x, _), (test_x, _), train_boxes, test_boxes, label_names = ret
    id_to_data = {
        'train_x': train_x,
        'test_x': test_x,
        'train_boxes': train_boxes,
        'test_boxes': test_boxes,
        'label_names': label_names
    }

    for data_str in ['train_x', 'test_x',                        \
                     'train_boxes', 'test_boxes', 'label_names']:
        np.save(cache_path[data_str], id_to_data[data_str])


def fetch_data():
    cache_dirname = os.path.join('datasets', 'tiny-imagenet-200')
    dataset_name = 'tiny-imagenet-200'
    base = 'http://cs231n.stanford.edu/'
    zipname = 'tiny-imagenet-200.zip'

    dataset_zippath = get_file(zipname,
                               origin=base+zipname,
                               cache_subdir=cache_dirname)
    dataset_dirname = os.path.dirname(dataset_zippath)
    dataset_subdir = os.path.join(dataset_dirname, dataset_name)

    # Extract zip file if it hasn't been extracted
    if not os.path.isdir(dataset_subdir):
        with zipfile.ZipFile(dataset_zippath, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(dataset_subdir))

    data_types = ['train_x', 'test_x', 'train_boxes', \
                  'test_boxes', 'label_names']
    cache_path = {}
    for data_type in data_types:
        cache_path[data_type] = os.path.join(os.path.dirname(dataset_subdir),
                                             'cached_' + data_type + '.npy')

    # If all caches are exist, load them
    if all([os.path.exists(cache_path[data_type]) for data_type in data_types]):
        return load_caches(cache_path)

    train_dir = os.path.join(dataset_subdir, 'train/')
    valid_dir = os.path.join(dataset_subdir, 'val/')

    num_classes = 200
    num_samples_train = 500
    num_samples_test = 50
    width, height, channels = 64, 64, 3

    # Create empty containers for dataset
    train_x = np.empty([num_classes*num_samples_train, width, height, channels],
                       dtype=np.uint8)

    train_y = np.array(np.arange(num_classes)                       \
                         .repeat(num_samples_train), dtype=np.uint8)\
                         .reshape((-1, 1))                          \

    train_boxes = np.empty([num_classes*num_samples_train, 4], dtype=np.uint8)

    test_x = np.empty([num_classes*num_samples_test, width, height, channels],
                      dtype=np.uint8)

    test_y = np.array(np.arange(num_classes)                        \
                        .repeat(num_samples_test), dtype=np.uint8)  \
                        .reshape((-1, 1))                           \

    test_boxes = np.empty([num_classes*num_samples_test, 4], dtype=np.uint8)


    # Build dictionary that maps label id to string
    label_by_id = {}
    with open(os.path.join(dataset_subdir, 'words.txt')) as word_file:
        lines = word_file.readlines()
    for i, line in enumerate(lines):
        label_id, name = lines[i].rstrip('\n').split('\t')
        label_by_id[label_id] = name


    # Process training dataset
    logging.info("Processing training data ..")

    label_ids = sorted(os.listdir(train_dir))
    label_names = np.array([label_by_id[label_id] for label_id in label_ids])

    # For every directory name that refers one of its labels,
    for i, label_id in enumerate(label_ids):
        box_annotation_filepath = os.path.join(train_dir,
                                               label_id,
                                               label_id + '_boxes.txt')

        with open(box_annotation_filepath, 'r') as boxfile:
            annotations = [x.rstrip('\n').split('\t')[1:] \
                           for x in boxfile.readlines()]  \

        for j in range(num_samples_train):
            train_boxes[i*num_samples_train + j, :] \
                = np.array(annotations[j], dtype=np.uint8)

        image_path_dirname = os.path.join(train_dir, label_id, 'images')
        for j, image_path in enumerate(os.listdir(image_path_dirname)):
            img_fullpath = os.path.join(image_path_dirname, image_path)
            image_arr = np.array(PIL.Image.open(img_fullpath), dtype=np.uint8)

            # If the image is grayscaled, span to RGB channels with same values
            if image_arr.shape == (width, height):
                image_arr = image_arr.repeat(channels) \
                                     .reshape(width, height, channels)

            train_x[(i * num_samples_train) + j, :, :, :] = image_arr


    # Process validation dataset
    logging.info("Processing test data ..")
    with open(os.path.join(valid_dir, 'val_annotations.txt')) as val_ann_file:
        val_annotations = [line.rstrip('\n').split('\t') \
                           for line in val_ann_file.readlines()]
        # Sort annotations by label id
        val_annotations = sorted(val_annotations, key=lambda x: x[1])

    for i, tup in enumerate(val_annotations):
        img_name, label_id, x0, y0, x1, y1 = tup

        test_boxes[i, :] = np.array([x0, y0, x1, y1], dtype=np.uint8)

        # Load, assign image to container
        img_fullpath = os.path.join(valid_dir, 'images', img_name)
        image_arr = np.array(PIL.Image.open(img_fullpath), dtype=np.uint8)

        if image_arr.shape == (width, height):
            image_arr = image_arr.repeat(channels) \
                                 .reshape(width, height, channels)

        test_x[i, :, :, :] = image_arr

    ret = (train_x, train_y), (test_x, test_y), \
          train_boxes, test_boxes, label_names  \

    save_caches(ret, cache_path)
    return ret


# Take 64x64 .npy arrays, preprocessing
def load_data(shape=(64, 64)):

    width, height = shape
    num_trainset = 100000
    num_testset = 10000

    # Duplicated with fetch_data(), will find alternatives.
    cache_dirname = os.path.join('datasets', 'tiny-imagenet-200')
    dataset_name = 'tiny-imagenet-200'
    base = 'http://cs231n.stanford.edu/'
    zipname = 'tiny-imagenet-200.zip'

    dataset_zippath = get_file(zipname,
                               origin=base+zipname,
                               cache_subdir=cache_dirname)
    dataset_dirname = os.path.dirname(dataset_zippath)
    dataset_subdir = os.path.join(dataset_dirname, dataset_name)


    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError("resize_shape must be 2d tuple\
                        consists of (width, heights)")
    if width > 64 or height > 64:
        raise ValueError("Cannot be resized bigger than the original image")

    cache_dir_train = os.path.join(os.path.dirname(dataset_subdir),
                                   "cached_{}x{}_train.npy"\
                                   .format(width, height)) \

    cache_dir_test = os.path.join(os.path.dirname(dataset_subdir),
                                  "cached_{}x{}_test.npy"  \
                                  .format(width, height))  \

    if (width, height) == (64, 64):
        cache_dir_train = os.path.join(os.path.dirname(cache_dir_train),
                                       'cached_train_x.npy')
        cache_dir_test = os.path.join(os.path.dirname(cache_dir_test),
                                      'cached_test_x.npy')

    (train_x, train_y), (test_x, test_y), _, _, _ = fetch_data()

    if os.path.exists(cache_dir_train) and os.path.exists(cache_dir_test):
        train_x = np.load(cache_dir_train)
        test_x = np.load(cache_dir_test)
        return (train_x, train_y), (test_x, test_y)


    resized_train = np.empty((num_trainset, width, height, 3), dtype='uint8')
    resized_test = np.empty((num_testset, width, height, 3), dtype='uint8')

    for i, image in enumerate(train_x):
        resized_train[i, :, :, :] = skimage.transform.resize(
            image,
            (width, height, 3),
            mode='constant',
            anti_aliasing=True,
            preserve_range=True)

    for i, image in enumerate(test_x):
        resized_test[i, :, :, :] = skimage.transform.resize(
            image,
            (width, height, 3),
            mode='constant',
            anti_aliasing=True,
            preserve_range=True)

    np.save(cache_dir_train, resized_train)
    np.save(cache_dir_test, resized_test)

    return (resized_train, train_y), (resized_test, test_y)
