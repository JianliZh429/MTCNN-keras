"""Use random crop to generate training data for three models."""
#
# MIT License
# Reference:
#    https://github.com/Seanlinx/mtcnn/blob/master/prepare_data/gen_pnet_data.py
#

import argparse
import os
import sys

import cv2
import numpy as np
import numpy.random as npr

from train.config import NET_SIZE
from train.utils import iou

CURR_DIR = os.path.dirname(__file__)


def main(args):
    net_name = args.net_name
    assert net_name in ['p_net', 'r_net', 'o_net']

    im_dir = args.wider_face_dir
    anno_file = args.annotation_file
    out_dir = args.out_dir

    target_size = NET_SIZE[net_name]
    save_dir = '{}/{}'.format(out_dir, net_name)

    pos_save_dir = save_dir + '/positive'
    part_save_dir = save_dir + '/part'
    neg_save_dir = save_dir + '/negative'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)
    if not os.path.exists(part_save_dir):
        os.mkdir(part_save_dir)
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)

    f1 = open(os.path.join(save_dir, 'pos_' + str(target_size) + '.txt'), 'w')
    f2 = open(os.path.join(save_dir, 'neg_' + str(target_size) + '.txt'), 'w')
    f3 = open(os.path.join(save_dir, 'part_' + str(target_size) + '.txt'), 'w')
    with open(anno_file, 'r') as f:
        annotations = f.readlines()
    num = len(annotations)
    print('%d pics in total' % num)
    p_idx = 0  # positive
    n_idx = 0  # negative
    d_idx = 0  # dont care
    idx = 0
    box_idx = 0
    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        im_path = annotation[0]
        bbox = list(map(float, annotation[1:]))
        boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
        img = cv2.imread(os.path.join(im_dir, im_path + '.jpg'))
        idx += 1
        if idx % 1000 == 0:
            print(idx, 'images done')

        height, width, channel = img.shape

        neg_num = 0
        while neg_num < 50:
            size = npr.randint(target_size, min(width, height) / 2)
            nx = npr.randint(0, width - size)
            ny = npr.randint(0, height - size)
            crop_box = np.array([nx, ny, nx + size, ny + size])

            _iou = iou(crop_box, boxes)

            cropped_im = img[ny: ny + size, nx: nx + size, :]
            resized_im = cv2.resize(cropped_im, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

            if np.max(_iou) < 0.3:
                # _iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, '%s.jpg' % n_idx)
                f2.write(save_dir + '/negative/%s' % n_idx + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
            print('{} images done, pos: {},  part: {},  neg: {}'.format(idx, p_idx, d_idx, n_idx))

        for box in boxes:
            x1, y1, x2, y2 = box
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            if max(w, h) < 40 or x1 < 0 or y1 < 0:
                continue

            # generate negative examples that have overlap with gt
            for i in range(5):
                size = npr.randint(target_size, min(width, height) / 2)
                # delta_x and delta_y are offsets of (x1, y1)
                delta_x = npr.randint(max(-size, -x1), w)
                delta_y = npr.randint(max(-size, -y1), h)
                nx1 = int(max(0, x1 + delta_x))
                ny1 = int(max(0, y1 + delta_y))
                if nx1 + size > width or ny1 + size > height:
                    continue
                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                _iou = iou(crop_box, boxes)

                cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
                resized_im = cv2.resize(cropped_im, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

                if np.max(_iou) < 0.3:
                    # _iou with all gts must below 0.3
                    save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                    f2.write(save_dir + "/negative/%s" % n_idx + ' 0\n')
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1

            # generate positive examples and part faces
            for i in range(20):
                size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

                delta_x = npr.randint(-w * 0.2, w * 0.2)
                delta_y = npr.randint(-h * 0.2, h * 0.2)

                nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
                nx2 = nx1 + size
                ny2 = ny1 + size

                if nx2 > width or ny2 > height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])

                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)

                cropped_im = img[ny1: ny2, nx1: nx2, :]
                resized_im = cv2.resize(cropped_im, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

                box_ = box.reshape(1, -1)
                if iou(crop_box, box_) >= 0.65:
                    save_file = os.path.join(pos_save_dir, '%s.jpg' % p_idx)
                    f1.write(save_dir + '/positive/%s' % p_idx + ' 1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))

                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                elif iou(crop_box, box_) >= 0.4:
                    save_file = os.path.join(part_save_dir, '%s.jpg' % d_idx)
                    f3.write(save_dir + '/part/%s' % d_idx + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
            box_idx += 1
            print('{} images done, pos: {},  part: {},  neg: {}'.format(idx, p_idx, d_idx, n_idx))

    f1.close()
    f2.close()
    f3.close()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('net_name', type=str, help='The specific net, p_net, r_net, or o_net')
    parser.add_argument('wider_face_dir', type=str, help='Directory for WIDER_FACE images')
    parser.add_argument('--annotation_file', type=str, help='WIDER_FACE Annotation file path for train data')
    parser.add_argument('--out_dir', type=str, default='.', help='Output dir to save generated dataset')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
