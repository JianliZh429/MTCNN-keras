import argparse
import os
import pickle
import random
import sys

import cv2
import numpy as np
from progress.bar import Bar

from train.config import NET_SIZE
from train.utils import resize


def main(args):
    net_name = args.net_name
    assert net_name in ['p_net', 'r_net', 'o_net']
    wider_face_dir = args.wider_face_dir
    landmark_dir = args.landmark_dir

    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    target_size = NET_SIZE[net_name]
    net_data_dir = os.path.join(wider_face_dir, net_name)

    with open('%s/pos_%s.txt' % (net_data_dir, target_size), 'r') as f:
        pos = f.readlines()
    with open('%s/neg_%s.txt' % (net_data_dir, target_size), 'r') as f:
        neg = f.readlines()
    with open('%s/part_%s.txt' % (net_data_dir, target_size), 'r') as f:
        part = f.readlines()
    with open('{}/landmarks_{}.txt'.format(landmark_dir, net_name), 'r', encoding='utf-8') as f:
        landmark_anno = f.readlines()

    create_label_dataset(net_name, neg, pos, target_size, out_dir)

    create_bbox_dataset(net_name, part, pos, target_size, out_dir)

    create_landmark_dataset(net_name, landmark_anno, target_size, out_dir)


def create_landmark_dataset(net_name, landmark_anno, target_size, out_dir):
    total_landmarks = len(landmark_anno)
    ims = []
    landmarks = []
    bar = Bar('Create landmark dataset', max=total_landmarks)
    for line in landmark_anno:
        bar.next()

        words = line.strip().split()
        image_file_name = words[0]

        im = cv2.imread(image_file_name)
        im = resize(im, target_size)
        im = im.astype('uint8')
        ims.append(im)

        landmark = words[2:12]
        landmark = list(map(float, landmark))

        landmarks.append(landmark)

    landmark_data = list(zip(ims, landmarks))
    random.shuffle(landmark_data)
    ims, landmarks = zip(*landmark_data)

    landmark_data_filename = os.path.join(out_dir, 'landmarks_{}.pkl'.format(net_name))

    with open(landmark_data_filename, 'wb') as f:
        pickle.dump({'ims': ims, 'landmarks': landmarks}, f)

    bar.finish()
    print('landmarks data done, total: {}'.format(len(ims)))


def create_bbox_dataset(net_name, part, pos, target_size, out_dir):
    total_pos = len(pos)
    total_part = len(part)
    ims = []
    labels = []
    bboxes = []
    bar = Bar('Create bbox dataset', max=total_pos + total_part)
    for line in pos:
        bar.next()

        words = line.split()
        image_file_name = words[0] + '.jpg'
        im = cv2.imread(image_file_name)
        im = resize(im, target_size)
        im = im.astype('uint8')
        ims.append(im)

        labels.append(np.array([0, 1]))

        box = np.array([float(words[2]), float(words[3]), float(words[4]), float(words[5])], dtype='float32')
        bboxes.append(box)

    part_stay = np.random.choice(len(part), size=200000, replace=False)
    for i in part_stay:
        bar.next()

        line = part[i]
        words = line.split()
        image_file_name = words[0] + '.jpg'
        im = cv2.imread(image_file_name)
        resize(im, target_size=target_size)
        im = im.astype('uint8')
        ims.append(im)

        labels.append(np.array([0, 0]))

        box = np.array([float(words[2]), float(words[3]), float(words[4]), float(words[5])], dtype='float32')
        bboxes.append(box)

    bboxes_data = list(zip(ims, labels, bboxes))
    random.shuffle(bboxes_data)
    ims, labels, bboxes = zip(*bboxes_data)
    label_data_filename = os.path.join(out_dir, 'bboxes_{}.pkl'.format(net_name))
    with open(label_data_filename, 'wb') as f:
        pickle.dump({'ims': ims, 'labels': labels, 'bboxes': bboxes}, f)

    bar.finish()
    print('bboxes data done, total: {}'.format(len(ims)))


def create_label_dataset(net_name, neg, pos, target_size, out_dir):
    total_pos = len(pos)
    total_neg = len(neg)
    ims = []
    labels = []
    bar = Bar('Create label dataset', max=total_pos + total_neg)
    for line in pos:
        bar.next()

        words = line.split()
        image_file_name = words[0] + '.jpg'
        im = cv2.imread(image_file_name)
        im = resize(im, target_size=target_size)
        im = im.astype(np.int8)
        ims.append(im)
        labels.append(np.array([0, 1]))

    neg_stay = np.random.choice(total_neg, size=600000, replace=False)
    for i in neg_stay:
        bar.next()

        line = neg[i]
        words = line.split()
        image_file_name = words[0] + '.jpg'

        im = cv2.imread(image_file_name)
        im = resize(im, target_size=target_size)
        im = im.astype(np.int8)
        ims.append(im)
        labels.append(np.array([1, 0]))

    label_data = list(zip(ims, labels))
    random.shuffle(label_data)
    ims, labels = zip(*label_data)
    label_data_filename = os.path.join(out_dir, 'label_{}.pkl'.format(net_name))
    with open(label_data_filename, 'wb') as f:
        pickle.dump({'ims': ims, 'labels': labels}, f)

    bar.finish()
    print('label data done, total : {}'.format(len(ims)))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('net_name', type=str, help='The specific net, p_net, r_net, or o_net')
    parser.add_argument('wider_face_dir', type=str, help='Directory for label and bounding box dataset')
    parser.add_argument('landmark_dir', type=str, help='Directory for landmark dataset')
    parser.add_argument('--out_dir', type=str, default='.', help='Output dir to save generated dataset')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
