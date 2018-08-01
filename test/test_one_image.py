import numpy as np
import sys
from argparse import ArgumentParser

import cv2

from mtcnn.detector import Detector


def main(weight_dir, image_file):
    detector = Detector(weight_dir=weight_dir, mode=1)
    im = cv2.imread(image_file)
    im = (im - 127.5) / 128
    labels, bboxes, landmarks = detector.predict([im])
    print(labels)
    print(bboxes)
    print(landmarks)
    for box in bboxes:
        box = np.squeeze(box)
        print('box is {}'.format(box))
        x1, y1, x2, y2 = box
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('image', im)
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('weight_dir', type=str, help='Directory of weights files')
    parser.add_argument('image_path', type=str, help='Image to test')
    args = parser.parse_args(sys.argv[1:])
    main(args.weight_dir, args.image_path)
