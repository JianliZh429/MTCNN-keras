import argparse
import os
import pickle
import sys

import cv2
import numpy as np
from progress.bar import Bar

from mtcnn.detector import Detector
from train.config import NET_SIZE, NET_NAMES
from train.utils import bbox_2_square, iou


def load_widerface_dataset(images_dir, anno_file):
    data = dict()
    images = []
    bboxes = []
    with open(anno_file, 'r', encoding='utf-8')as f:
        while True:
            line = f.readline().strip()
            if not line:
                break
            image_path = line
            images.append(os.path.join(images_dir, image_path))
            face_num = int(f.readline().strip())
            faces = []
            for i in range(face_num):
                bb_info = f.readline().strip('\n').split(' ')
                x1, y1, w, h = [float(bb_info[i]) for i in range(4)]
                faces.append([x1, y1, x1 + w, y1 + h])
            bboxes.append(faces)
        data['images'] = images  # all image pathes
        data['bboxes'] = bboxes  # all image bboxes

    return data


def build_save_path(out_dir):
    neg_dir = os.path.join(out_dir, 'negative')
    pos_dir = os.path.join(out_dir, 'positive')
    part_dir = os.path.join(out_dir, 'part')
    for file_dir in [neg_dir, pos_dir, part_dir]:
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
    return neg_dir, pos_dir, part_dir


def save_hard_examples(net_name, out_dir, dataset, detections_path):
    neg_dir, pos_dir, part_dir = build_save_path(out_dir)
    neg_file = open(os.path.join(out_dir, 'neg_{}.txt'.format(net_name)), 'w', encoding='utf-8')
    pos_file = open(os.path.join(out_dir, 'pos_{}.txt'.format(net_name)), 'w', encoding='utf-8')
    part_file = open(os.path.join(out_dir, 'part_{}.txt'.format(net_name)), 'w', encoding='utf-8')

    with open(detections_path, 'rb') as f:
        detections = pickle.load(f)

    image_files = dataset['images']
    bboxes_true = dataset['bboxes']
    bboxes_pred = detections['bboxes']
    bar = Bar('Save hard examples', max=len(image_files))
    im_size = NET_SIZE[net_name]
    n_idx = 0
    p_idx = 0
    d_idx = 0

    for im_f, bbox_pred, box_true in zip(image_files, bboxes_pred, bboxes_true):
        bar.next()
        if bbox_pred.shape[0] == 0:
            continue
        box_true = np.array(box_true, dtype=np.float32).reshape(-1, 4)
        img = cv2.imread(im_f)
        bbox_pred = bbox_2_square(bbox_pred)
        bbox_pred[:, 0:4] = np.round(bbox_pred[:, 0:4])

        neg_num = 0
        for box in bbox_pred:
            x1, y1, x2, y2, _ = box
            width = x2 - x1 + 1
            height = y2 - y1 + 1

            if width < 20 or x1 < 0 or y1 < 0 or x2 > img.shape[1] - 1 or y2 > img.shape[0] - 1:
                continue

            _iou = iou(box, box_true)
            print('box: {}, box_true: {}, _iou: {}'.format(box, box_true, _iou))
            cropped_im = img[int(y1):int(y2 + 1), int(x1):int(x2 + 1), :]
            resized_im = cv2.resize(cropped_im, (im_size, im_size), interpolation=cv2.INTER_LINEAR)
            if np.max(_iou) < 0.3 and neg_num < 60:
                file_name = os.path.join(neg_dir, '{0:08}.jpg'.format(n_idx))
                cv2.imwrite(file_name, resized_im)
                neg_file.write('{} 0\n'.format(file_name))
                n_idx += 1
                neg_num += 1
            else:
                idx = np.argmax(_iou)
                assigned_gt = box_true[idx]
                x1t, y1t, x2t, y2t = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x1t) / float(width)
                offset_y1 = (y1 - y1t) / float(height)
                offset_x2 = (x2 - x2t) / float(width)
                offset_y2 = (y2 - y2t) / float(height)

                if np.max(_iou) >= 0.65:
                    file_name = os.path.join(neg_dir, '{0:08}.jpg'.format(p_idx))
                    pos_file.write(
                        file_name + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(file_name, resized_im)
                    p_idx += 1

                elif np.max(_iou) >= 0.4:
                    file_name = os.path.join(neg_dir, '{0:08}.jpg'.format(d_idx))
                    part_file.write(file_name + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(file_name, resized_im)
                    d_idx += 1
    neg_file.close()
    part_file.close()
    pos_file.close()

    bar.finish()


def main(args):
    net_name = args.net_name
    images_dir = args.wider_face_dir
    annotation_file = args.annotation_file
    out_dir = args.out_dir

    assert net_name in NET_NAMES
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    detections_path = os.path.join(out_dir, 'detections.pkl')

    dataset = load_widerface_dataset(images_dir, annotation_file)
    if not os.path.exists(detections_path):
        detector = Detector(weight_dir=args.weights, mode=1)
        length = len(dataset['images'])
        bar = Bar(message='Load to np images...', max=length)
        np_images = []
        for img in dataset['images']:
            bar.next()
            im = cv2.imread(img)
            np_images.append(im)
        bar.finish()

        bboxes, landmarks = detector.predict(np_images, verbose=True)

        with open(detections_path, 'wb') as f:
            pickle.dump({
                'bboxes': bboxes,
                'landmarks': landmarks
            }, f)

    save_hard_examples(net_name, out_dir, dataset, detections_path)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('net_name', type=str, help='The specific net, p_net, r_net, or o_net')
    parser.add_argument('wider_face_dir', type=str, help='Directory for WIDER_FACE images')
    parser.add_argument('--weights', type=str, help='Weights')
    parser.add_argument('--annotation_file', type=str, help='WIDER_FACE Annotation file path for train data')
    parser.add_argument('--out_dir', type=str, default='.', help='Output dir to save generated dataset')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
