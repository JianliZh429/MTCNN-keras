import pickle

import numpy as np


def _load_dataset(dataset_path):
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def load_label_dataset(label_dataset_path):
    label_dataset = _load_dataset(label_dataset_path)
    label_x = label_dataset['ims']
    label_y = label_dataset['labels']
    label_x = np.array(label_x)
    label_y = np.array(label_y)
    return label_x, label_y


def load_bbox_dataset(bbox_dataset_path):
    bbox_dataset = _load_dataset(bbox_dataset_path)
    bbox_x = bbox_dataset['ims']
    bbox_y = bbox_dataset['bboxes']
    bbox_x = np.array(bbox_x)
    bbox_y = np.array(bbox_y)

    return bbox_x, bbox_y


def load_landmark_dataset(landmark_dataset_path):
    landmark_dataset = _load_dataset(landmark_dataset_path)

    landmark_x = landmark_dataset['ims']
    landmark_y = landmark_dataset['landmarks']
    landmark_x = np.array(landmark_x)
    landmark_y = np.array(landmark_y)
    return landmark_x, landmark_y
