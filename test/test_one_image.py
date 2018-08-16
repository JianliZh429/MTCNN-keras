import sys
from argparse import ArgumentParser

import cv2

from mtcnn.detector import Detector


def main(weight_dir, image_file):
    detector = Detector(weight_dir=weight_dir, mode=3, min_face_size=24)
    im = cv2.imread(image_file)
    bboxes, landmarks = detector.predict([im])
    print('------------------{}------------'.format(len(bboxes)))
    bboxes = bboxes[0]
    print('Faces: {}'.format(len(bboxes)))
    for box in bboxes[:10]:
        print('box is {}'.format(box))
        x1, y1, x2, y2, _ = box
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('image', im)
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('weight_dir', type=str, help='Directory of weights files')
    parser.add_argument('image_path', type=str, help='Image to test')
    args = parser.parse_args(sys.argv[1:])
    main(args.weight_dir, args.image_path)
