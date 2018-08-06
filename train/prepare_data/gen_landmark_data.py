# coding: utf-8
import os
import random
import sys
from argparse import ArgumentParser

import cv2
import numpy as np
import numpy.random as npr

from train.config import NET_SIZE, NET_NAMES
from train.utils import resize, iou, flip, rotate


def read_cnn_face_points(dataset_dir):
    landmark_annotation_file = os.path.join(dataset_dir, 'trainImageList.txt')
    data = []
    with open(landmark_annotation_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            splits = line.split(' ')
            image_file = splits[0]
            image_file = image_file.replace('\\', os.sep)
            bbox = list(map(int, splits[1:5]))
            bbox = bbox[0], bbox[2], bbox[1], bbox[3]
            landmark = [(float(splits[5 + 2 * i]), float(splits[5 + 2 * i + 1])) for i in range(0, 5)]

            data.append({
                'image_file': os.path.join(dataset_dir, image_file),
                'bbox': bbox,
                'landmark': landmark
            })
    return data


def generate_data(net_name, dataset_dir, out_dir, augmentation=False):
    assert net_name in NET_NAMES
    size = NET_SIZE[net_name]

    data = read_cnn_face_points(dataset_dir)
    face_images, face_landmarks = process_data(data, size, augmentation)
    save_data(face_images, face_landmarks, net_name, out_dir)


def process_data(data, size, augmentation):
    face_images = []
    face_landmarks = []
    idx = 0
    for d in data:
        image_file = d['image_file']
        bbox = d['bbox']
        points = d['landmark']

        img = cv2.imread(image_file)
        im_h, im_w, _ = img.shape
        x1, y1, x2, y2 = bbox
        face_roi = img[y1:y2, x1:x2]
        face_roi = resize(face_roi, size)

        # normalization
        landmark = normalize_landmark(points, x1, x2, y1, y2)
        face_images.append(face_roi)
        face_landmarks.append(np.array(landmark).reshape(10))

        if augmentation:
            idx = idx + 1
            if idx % 100 == 0:
                print(idx, "images done")
            # gt's width
            bbox_w = x2 - x1 + 1
            # gt's height
            bbox_h = y2 - y1 + 1
            if max(bbox_w, bbox_h) < 40 or x1 < 0 or y1 < 0:
                continue
            # random shift
            for i in range(10):
                bbox_size, nx1, nx2, ny1, ny2 = new_bbox(bbox_h, bbox_w, x1, y1)
                if nx2 > im_w or ny2 > im_h:
                    continue

                crop_box = np.array([nx1, ny1, nx2, ny2])
                cropped_im = img[ny1:ny2 + 1, nx1:nx2 + 1, :]

                _iou = iou(crop_box, np.expand_dims(bbox, 0))

                if _iou > 0.65:
                    resize_im = resize(cropped_im, size)
                    face_images.append(resize_im)
                    # normalize
                    landmark = normalize_landmark2(bbox_size, points, nx1, ny1)
                    face_landmarks.append(landmark.reshape(10))

                    landmark_ = landmark.reshape(-1, 2)

                    nbbox = nx1, ny1, nx2, ny2

                    # mirror
                    if random.choice([0, 1]) > 0:
                        face_flipped, landmark_flipped = flip(resize_im, landmark_)
                        face_flipped = resize(face_flipped, size)
                        # c*h*w
                        face_images.append(face_flipped)
                        face_landmarks.append(landmark_flipped.reshape(10))
                    # rotate
                    if random.choice([0, 1]) > 0:
                        _landmark = reproject_landmark(nbbox, landmark_)
                        face_rotated_by_alpha, landmark_rotated = rotate(img, nbbox, _landmark, 5)
                        # landmark_offset
                        landmark_rotated = project_landmark(nbbox, landmark_rotated)

                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        face_images.append(face_rotated_by_alpha)
                        face_landmarks.append(landmark_rotated.reshape(10))

                        # flip
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        face_images.append(face_flipped)
                        face_landmarks.append(landmark_flipped.reshape(10))

                        # inverse clockwise rotation
                    if random.choice([0, 1]) > 0:
                        _landmark = reproject_landmark(nbbox, landmark_)
                        face_rotated_by_alpha, landmark_rotated = rotate(img, nbbox, _landmark, -5)  # 顺时针旋转

                        landmark_rotated = project_landmark(nbbox, landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        face_images.append(face_rotated_by_alpha)
                        face_landmarks.append(landmark_rotated.reshape(10))

                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        face_images.append(face_flipped)
                        face_landmarks.append(landmark_flipped.reshape(10))

    return face_images, face_landmarks


def project(bbox, point):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    x = (point[0] - x1) / w
    y = (point[1] - y1) / h
    return np.asarray([x, y])


def project_landmark(bbox, landmark):
    p = np.zeros((len(landmark), 2))
    for i in range(len(landmark)):
        p[i] = project(bbox, landmark[i])
    return p


def reproject(bbox, point):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    x = x1 + w * point[0]
    y = y1 + h * point[1]
    return np.asarray([x, y])


def reproject_landmark(bbox, landmark):
    p = np.zeros((len(landmark), 2))
    for i in range(len(landmark)):
        p[i] = reproject(bbox, landmark[i])
    return p


def normalize_landmark2(bbox_size, points, x1, y1):
    landmark = np.zeros((5, 2))
    for i, point in enumerate(points):
        rv = ((point[0] - x1) / bbox_size, (point[1] - y1) / bbox_size)
        landmark[i] = rv
    return landmark


def normalize_landmark(points, x1, x2, y1, y2):
    landmark = np.zeros((5, 2))
    for i, point in enumerate(points):
        rv = (point[0] - x1) / (x2 - x1), (point[1] - y1) / (y2 - y1)
        landmark[i] = rv
    return landmark


def new_bbox(bbox_h, bbox_w, x1, y1):
    bbox_size = npr.randint(int(min(bbox_w, bbox_h) * 0.8), np.ceil(1.25 * max(bbox_w, bbox_h)))
    delta_x = npr.randint(-bbox_w * 0.2, bbox_w * 0.2)
    delta_y = npr.randint(-bbox_h * 0.2, bbox_h * 0.2)
    nx1 = int(max(x1 + bbox_w / 2 - bbox_size / 2 + delta_x, 0))
    ny1 = int(max(y1 + bbox_h / 2 - bbox_size / 2 + delta_y, 0))
    nx2 = nx1 + bbox_size
    ny2 = ny1 + bbox_size
    return bbox_size, nx1, nx2, ny1, ny2


def save_data(face_images, face_landmarks, net_name, out_dir):
    im_count = 0
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    img_dir = os.path.join(out_dir, 'landmarks_{}'.format(net_name))
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    output_file = os.path.join(out_dir, 'landmarks_{}.txt'.format(net_name))
    with open(output_file, 'w', encoding='utf-8') as f:
        for im, point in zip(face_images, face_landmarks):
            if np.sum(np.where(point <= 0, 1, 0)) > 0:
                continue

            if np.sum(np.where(point >= 1, 1, 0)) > 0:
                continue
            im_f = os.path.join(img_dir, '{0:08d}.jpg'.format(im_count))
            print('processing {}'.format(im_f))
            cv2.imwrite(im_f, im)

            txt = '{} -2 {}\n'.format(im_f, ' '.join(map(str, list(point))))
            f.write(txt)

            im_count += 1


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('net_name', type=str, help='Net name, should be p_net, r_net or o_net')
    parser.add_argument('dataset_dir', type=str, help='Directory to dataset')
    parser.add_argument('--out_dir', type=str, default='.', help='Where to save generated data')
    parser.add_argument('--augmentation', type=bool, default=True, help='If use augmentation')
    args = parser.parse_args(sys.argv[1:])
    generate_data(args.net_name, args.dataset_dir, args.out_dir, args.augmentation)
