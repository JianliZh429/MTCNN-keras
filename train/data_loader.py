import pickle

import numpy as np
from keras.utils import to_categorical


def _load_pickle(dataset_path):
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def _process_im(im):
    return (im.astype(np.float32) - 127.5) / 128


label_map = {'0': [0, 1], '1': [1, 0], '-1': [0, 0], '-2': [1, 1]}


def _process_label(labels):
    label = []
    for ll in labels:
        label.append(label_map.get(str(ll)))
    return label


def load_dataset(label_dataset_path, bbox_dataset_path, landmark_dataset_path, im_size=12):
    images_x = np.empty((0, im_size, im_size, 3))
    labels_y = np.empty((0, 2))
    bboxes_y = np.empty((0, 4))
    landmarks_y = np.empty((0, 10))

    label_x, label_y = load_label_dataset(label_dataset_path)
    len_labels = len(label_y)
    images_x = np.concatenate((images_x, label_x), axis=0)
    labels_y = np.concatenate((labels_y, label_y), axis=0)
    bboxes_y = np.concatenate((bboxes_y, np.zeros((len_labels, 4), np.float32)), axis=0)
    landmarks_y = np.concatenate((landmarks_y, np.zeros((len_labels, 10), np.float32)), axis=0)

    bbox_x, bbox_y, b_label_y = load_bbox_dataset(bbox_dataset_path)
    len_labels = len(b_label_y)
    images_x = np.concatenate((images_x, bbox_x), axis=0)
    labels_y = np.concatenate((labels_y, b_label_y), axis=0)
    bboxes_y = np.concatenate((bboxes_y, bbox_y), axis=0)
    landmarks_y = np.concatenate((landmarks_y, np.zeros((len_labels, 10), np.float32)), axis=0)

    landmark_x, landmark_y, l_label_y = load_landmark_dataset(landmark_dataset_path)
    len_labels = len(l_label_y)
    images_x = np.concatenate((images_x, landmark_x), axis=0)
    labels_y = np.concatenate((labels_y, l_label_y), axis=0)
    bboxes_y = np.concatenate((bboxes_y, np.array([[0, 0, im_size - 1, im_size - 1]] * len_labels, np.float32)), axis=0)
    landmarks_y = np.concatenate((landmarks_y, landmark_y), axis=0)

    assert len(images_x) == len(labels_y) == len(bboxes_y) == len(landmarks_y)

    print('Shape of all: \n')
    print(images_x.shape)
    print(labels_y.shape)
    print(bboxes_y.shape)
    print(landmarks_y.shape)

    return images_x, labels_y, bboxes_y, landmarks_y


def load_label_dataset(label_dataset_path):
    label_dataset = _load_pickle(label_dataset_path)
    label_x = label_dataset['ims']
    label_y = label_dataset['labels']

    label_x = _process_im(np.array(label_x))

    label_y = _process_label(label_y)

    label_y = np.array(label_y).astype(np.int8)
    return label_x, label_y


def load_bbox_dataset(bbox_dataset_path):
    bbox_dataset = _load_pickle(bbox_dataset_path)
    bbox_x = bbox_dataset['ims']
    bbox_y = bbox_dataset['bboxes']
    label_y = bbox_dataset['labels']
    bbox_x = _process_im(np.array(bbox_x))
    bbox_y = np.array(bbox_y).astype(np.float32)

    label_y = _process_label(label_y)
    label_y = np.array(label_y).astype(np.int8)

    return bbox_x, bbox_y, label_y


def load_landmark_dataset(landmark_dataset_path):
    landmark_dataset = _load_pickle(landmark_dataset_path)

    landmark_x = landmark_dataset['ims']
    landmark_y = landmark_dataset['landmarks']
    label_y = landmark_dataset['labels']

    landmark_x = _process_im(np.array(landmark_x))
    landmark_y = np.array(landmark_y).astype(np.float32)

    label_y = _process_label(label_y)
    label_y = np.array(label_y).astype(np.int8)

    return landmark_x, landmark_y, label_y
