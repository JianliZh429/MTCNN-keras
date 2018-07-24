import argparse
import os
import pickle
import random
import sys

import cv2
import numpy as np

from train.config import net_size


def resize(im, target_size):
    h, w, ch = im.shape
    if h != target_size or w != target_size:
        im = cv2.resize(im, (target_size, target_size))
    return im


def main(args):
    net_name = args.net_name
    assert net_name in ['p_net', 'r_net', 'o_net']
    base_dir = args.base_dir
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    target_size = net_size[net_name]
    net_data_dir = os.path.join(base_dir, net_name)
    with open('%s/pos_%s.txt' % (net_data_dir, target_size), 'r') as f:
        pos = f.readlines()
    with open('%s/neg_%s.txt' % (net_data_dir, target_size), 'r') as f:
        neg = f.readlines()
    with open('%s/part_%s.txt' % (net_data_dir, target_size), 'r') as f:
        part = f.readlines()

    print('Create label dataset.......')
    create_label_dataset(net_name, neg, pos, target_size, out_dir)
    create_bbox_dataset(net_name, part, pos, target_size, out_dir)


def create_bbox_dataset(net_name, part, pos, target_size, out_dir):
    total_pos = len(pos)
    print('Total positive is {}'.format(total_pos))
    ims = []
    labels = []
    bboxes = []
    for line in pos:
        words = line.split()
        image_file_name = words[0] + '.jpg'
        im = cv2.imread(image_file_name)
        im = resize(im, target_size)
        im = im.astype('uint8')
        ims.append(im)

        labels.append(np.array([0, 1]))

        box = np.array([float(words[2]), float(words[3]), float(words[4]), float(words[5])], dtype='float32')
        bboxes.append(box)
    total_part = len(part)
    print('Total partial is {}'.format(total_part))
    part_stay = np.random.choice(len(part), size=200000, replace=False)
    for i in part_stay:
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
    print('bboxes data length: {}'.format(len(ims)))
    bboxes_data = list(zip(ims, labels, bboxes))
    random.shuffle(bboxes_data)
    ims, labels, bboxes = zip(*bboxes_data)
    label_data_filename = os.path.join(out_dir, 'bboxes_{}.pkl'.format(net_name))
    with open(label_data_filename, 'wb') as f:
        pickle.dump({'ims': ims, 'labels': labels, 'bboxes': bboxes}, f)


def create_label_dataset(net_name, neg, pos, target_size, out_dir):
    total_pos = len(pos)
    print('Total positive is {}'.format(total_pos))
    ims = []
    labels = []
    for line in pos:
        words = line.split()
        image_file_name = words[0] + '.jpg'
        im = cv2.imread(image_file_name)
        im = resize(im, target_size=target_size)
        im = im.astype(np.int8)
        ims.append(im)
        labels.append(np.array([0, 1]))
    total_neg = len(neg)
    print('Total negative is {}'.format(total_neg))
    neg_stay = np.random.choice(total_neg, size=600000, replace=False)
    for i in neg_stay:
        line = neg[i]
        words = line.split()
        image_file_name = words[0] + '.jpg'

        im = cv2.imread(image_file_name)
        im = resize(im, target_size=target_size)
        im = im.astype(np.int8)
        ims.append(im)
        labels.append(np.array([1, 0]))

    print('label data length: {}'.format(len(ims)))
    label_data = list(zip(ims, labels))
    random.shuffle(label_data)
    ims, labels = zip(*label_data)
    label_data_filename = os.path.join(out_dir, 'label_{}.pkl'.format(net_name))
    with open(label_data_filename, 'wb') as f:
        pickle.dump({'ims': ims, 'labels': labels}, f)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('net_name', type=str, help='The specific net, p_net, r_net, or o_net')
    parser.add_argument('base_dir', type=str, help='Directory for data base dir')
    parser.add_argument('--out_dir', type=str, default='.', help='Output dir to save generated dataset')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
