import os
import pickle
import random

import cv2
import h5py
import numpy as np
import numpy.random as npr

from train.h5py_utils import load_dict_from_hdf5
from .config import LABEL_MAP


class DataGenerator:

    def __init__(self, label_dataset_path, bboxes_dataset_path, landmarks_dataset_path, batch_size, im_size):
        self.im_size = im_size
        self.label_file = h5py.File(label_dataset_path, 'r')
        self.bbox_file = h5py.File(bboxes_dataset_path, 'r')
        self.landmark_file = h5py.File(landmarks_dataset_path, 'r')

        self.batch_size = batch_size

        self.label_len = len(self.label_file['labels'])
        self.bbox_len = len(self.bbox_file['labels'])
        self.landmark_len = len(self.landmark_file['labels'])
        self.label_cursor = 0
        self.bbox_cursor = 0
        self.landmark_cursor = 0

        self.steps_per_epoch = int((self.label_len + self.bbox_len + self.landmark_len) / self.batch_size)

        self.index = 0
        self.epoch_done = False
        self._gen_sub_batch()

    def _reset(self):
        self.index = 0
        self.epoch_done = False
        self.label_cursor = 0
        self.bbox_cursor = 0
        self.landmark_cursor = 0

    def _gen_sub_batch(self):
        n = self.steps_per_epoch
        self.label_batch_list = npr.multinomial(self.label_len, np.ones(n) / n, size=1)[0]
        self.bbox_batch_list = npr.multinomial(self.bbox_len, np.ones(n) / n, size=1)[0]
        self.landmark_batch_list = npr.multinomial(self.landmark_len, np.ones(n) / n, size=1)[0]

    def _load_label_dataset(self):
        end = self.label_cursor + self.label_batch_list[self.index]

        im_batch = self.label_file['ims'][self.label_cursor:end]
        labels_batch = self.label_file['labels'][self.label_cursor:end]

        batch_size = im_batch.shape[0]

        bboxes_batch = np.zeros(shape=(batch_size, 4), dtype=np.float32)
        landmarks_batch = np.zeros(shape=(batch_size, 10), dtype=np.float32)

        self.label_cursor = end
        self.epoch_done = True if self.label_cursor >= self.label_len else self.epoch_done
        return im_batch, labels_batch, bboxes_batch, landmarks_batch

    def _load_bbox_dataset(self):
        end = self.bbox_cursor + self.bbox_batch_list[self.index]

        im_batch = self.bbox_file['ims'][self.bbox_cursor:end]
        box_batch = self.bbox_file['bboxes'][self.bbox_cursor:end]
        label_batch = self.bbox_file['labels'][self.bbox_cursor:end]

        batch_size = im_batch.shape[0]
        landmarks_batch = np.zeros(shape=(batch_size, 10), dtype=np.float32)

        self.bbox_cursor = end
        self.epoch_done = True if self.bbox_cursor >= self.bbox_len else self.epoch_done
        return im_batch, label_batch, box_batch, landmarks_batch

    def _load_landmark_dataset(self):
        end = self.landmark_cursor + self.landmark_batch_list[self.index]

        im_batch = self.landmark_file['ims'][self.landmark_cursor:end]
        landmark_batch = self.landmark_file['landmarks'][self.landmark_cursor:end]
        label_batch = self.landmark_file['labels'][self.landmark_cursor:end]

        batch_size = im_batch.shape[0]
        bboxes_batch = np.array([[0, 0, self.im_size - 1, self.im_size - 1]] * batch_size, np.float32)

        self.landmark_cursor = end
        self.epoch_done = True if self.landmark_cursor >= self.landmark_len else self.epoch_done
        return im_batch, label_batch, bboxes_batch, landmark_batch,

    def im_show(self, n):
        ns = random.sample(range(0, len(self.landmark_file['ims'][:])), n)
        for i in ns:
            im = self.landmark_file['ims'][i]
            cv2.imshow('{}'.format(i), im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def generate(self):
        while True:
            if self.index >= self.steps_per_epoch or self.epoch_done:
                self._reset()
                self._gen_sub_batch()

            im_batch1, labels_batch1, bboxes_batch1, landmarks_batch1 = self._load_label_dataset()
            im_batch2, labels_batch2, bboxes_batch2, landmarks_batch2 = self._load_bbox_dataset()
            im_batch3, labels_batch3, bboxes_batch3, landmarks_batch3 = self._load_landmark_dataset()
            self.index += 1

            x_batch = np.concatenate((im_batch1, im_batch2, im_batch3), axis=0)
            x_batch = _process_im(x_batch)
            if x_batch.shape[0] < 1:
                self.epoch_done = True
                continue

            label_batch = np.concatenate((labels_batch1, labels_batch2, labels_batch3), axis=0)
            label_batch = np.array(_process_label(label_batch))
            bbox_batch = np.concatenate((bboxes_batch1, bboxes_batch2, bboxes_batch3), axis=0)
            landmark_batch = np.concatenate((landmarks_batch1, landmarks_batch2, landmarks_batch3), axis=0)

            y_batch = np.concatenate((label_batch, bbox_batch, landmark_batch), axis=1)
            yield x_batch, y_batch

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.label_file.close()
        self.bbox_file.close()
        self.landmark_file.close()


def _load_dataset(dataset_path):
    ext = dataset_path.split(os.extsep)[-1]
    if ext == 'pkl':
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
    elif ext == 'h5':
        dataset = load_dict_from_hdf5(dataset_path)
    else:
        raise ValueError('Unsupported file type, only *.pkl and *.h5 are supported now.')
    return dataset


def _process_im(im):
    return (im.astype(np.float32) - 127.5) / 128


def _process_label(labels):
    label = []
    for ll in labels:
        label.append(LABEL_MAP.get(str(ll)))
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
    label_dataset = _load_dataset(label_dataset_path)
    label_x = label_dataset['ims']
    label_y = label_dataset['labels']

    label_x = _process_im(np.array(label_x))

    label_y = _process_label(label_y)

    label_y = np.array(label_y).astype(np.int8)
    return label_x, label_y


def load_bbox_dataset(bbox_dataset_path):
    bbox_dataset = _load_dataset(bbox_dataset_path)
    bbox_x = bbox_dataset['ims']
    bbox_y = bbox_dataset['bboxes']
    label_y = bbox_dataset['labels']
    bbox_x = _process_im(np.array(bbox_x))
    bbox_y = np.array(bbox_y).astype(np.float32)

    label_y = _process_label(label_y)
    label_y = np.array(label_y).astype(np.int8)

    return bbox_x, bbox_y, label_y


def load_landmark_dataset(landmark_dataset_path):
    landmark_dataset = _load_dataset(landmark_dataset_path)

    landmark_x = landmark_dataset['ims']
    landmark_y = landmark_dataset['landmarks']
    label_y = landmark_dataset['labels']

    landmark_x = _process_im(np.array(landmark_x))
    landmark_y = np.array(landmark_y).astype(np.float32)

    label_y = _process_label(label_y)
    label_y = np.array(label_y).astype(np.int8)

    return landmark_x, landmark_y, label_y
