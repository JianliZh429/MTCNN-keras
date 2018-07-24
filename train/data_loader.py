import json
import os
import pickle
import random
import sys
from argparse import ArgumentParser
from collections import Iterator

from keras.applications import ResNet50
from keras.preprocessing import image

image_ext = ['jpg', 'jpeg', 'png', 'gif']


def is_image(file_name):
    f_n = file_name.lower()
    seps = f_n.split(os.extsep)
    if len(seps) > 1 and seps[-1] in image_ext:
        return True
    return False
ResNet50

def convert_annotation_file(anno_file, output, suffix='train'):
    with open(anno_file, 'r', encoding='utf-8') as f:
        line = f.readline()
        image_name = None
        face_num = 0
        images = []
        while line:
            line = line.strip()
            faces = []
            if is_image(line):
                image_name = line
                face_num = int(f.readline())
                bboxes = []
                for i in range(0, face_num):
                    bboxes.append(f.readline())
                for box in bboxes:
                    x, y, w, h = box.split(' ')[:4]
                    x = int(x)
                    y = int(y)
                    w = int(w)
                    h = int(h)
                    faces.append([x, y, x + w, y + h])
            else:
                print('Cannot pass {} file, of {}'.format(anno_file, line))

            img = {
                'file_name': image_name,
                'face_num': face_num,
                'bboxes': faces
            }
            images.append(img)
            line = f.readline()
        if not os.path.exists(output):
            os.makedirs(output)
        with open(os.path.join(output, 'annotation_{}.json'.format(suffix)), 'w', encoding='utf-8') as file:
            json.dump(images, file)


def create_annotation_wider_face(train_anno_file, test_anno_file, output):
    if train_anno_file is not None:
        convert_annotation_file(train_anno_file, output, 'train')
    if test_anno_file is not None:
        convert_annotation_file(test_anno_file, output, 'val')


class WiderFaceDataset(Iterator):
    def __init__(self, images_dir, annotation_file, batch_size, shuffle=False, target_size=12):
        self.images_dir = images_dir
        self.annotation_file = annotation_file
        self.batch_size = batch_size
        self.curr = 0
        self.annotations = []
        self.target_size = target_size

        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        self.total = len(self.annotations)
        print('len annotations {}'.format(len(self.annotations)))
        if shuffle:
            random.shuffle(self.annotations)

    def __next__(self):
        end = self.curr + self.batch_size
        print('Curr: {}'.format(self.curr))
        if self.curr > self.total:
            raise StopIteration
        annotations = self.annotations[self.curr:end]

        x = []
        y = []
        y0 = []
        y1 = []
        for anno in annotations:
            filename = os.path.join(self.images_dir, anno['file_name'])
            im = image.load_img(filename, target_size=(self.target_size, self.target_size))
            im = image.img_to_array(im)
            im = im / 255
            x.append(im)
            y0.append(1)

            bboxes = anno['bboxes']
            y1.append(bboxes)

        self.curr = end
        y.append(y0)
        y.append(y1)

        return x, y


def read_images(train_dataset):
    train_x = []
    train_y = []
    y0 = []
    y1 = []
    count = 0
    for t in train_dataset:
        batch_x, batch_y = t
        train_x += batch_x
        y0 += batch_y[0]
        y1 += batch_y[1]
        print('Load dataset of batch: {}'.format(count))
        count += 1
    train_y.append(y0)
    train_y.append(y1)
    return train_x, train_y


def to_p_net_data(image_dir, output_file):
    train_annotation_file = os.path.join(image_dir, 'annotation_train.json')
    val_annotation_file = os.path.join(image_dir, 'annotation_val.json')

    train_dataset = WiderFaceDataset('{}/train'.format(image_dir), train_annotation_file, 1000, target_size=12)
    val_dataset = WiderFaceDataset('{}/val'.format(image_dir), val_annotation_file, 1000, target_size=12)

    train_x, train_y = read_images(train_dataset)
    val_x, val_y = read_images(val_dataset)

    with open(output_file, 'wb') as f:
        pickle.dump({
            'train_x': train_x,
            'train_y': train_y,
            'val_x': val_x,
            'val_y': val_y
        }, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('image_dir', type=str, help='Folder of data base dir')
    parser.add_argument('--output', type=str, default='wider_face.pkl', help='Output file name')

    args = parser.parse_args(sys.argv[1:])

    to_p_net_data(args.image_dir, args.output)
