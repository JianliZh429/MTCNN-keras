import cv2
import numpy as np


def iou(box, boxes):
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr


def resize(im, target_size):
    h, w, ch = im.shape
    if h != target_size or w != target_size:
        im = cv2.resize(im, (target_size, target_size))
    return im


def flip(face, landmark):
    face_flipped_by_x = cv2.flip(face, 1)
    landmark_ = np.asarray([(1 - x, y) for (x, y) in landmark])
    landmark_[[0, 1]] = landmark_[[1, 0]]
    landmark_[[3, 4]] = landmark_[[4, 3]]
    return face_flipped_by_x, landmark_


def rotate(img, bbox, landmark, alpha):
    """
        given a face with bbox and landmark, rotate with alpha
        and return rotated face with bbox, landmark (absolute position)
    """
    x1, y1, x2, y2 = bbox
    center = ((x1 + x2) / 2, (y1 + y2) / 2)

    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)

    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))

    landmark_ = np.asarray([(rot_mat[0][0] * x + rot_mat[0][1] * y + rot_mat[0][2],
                             rot_mat[1][0] * x + rot_mat[1][1] * y + rot_mat[1][2]) for (x, y) in landmark])
    face = img_rotated_by_alpha[y1:y2 + 1, x1:x2 + 1]
    return face, landmark_


def bbox_2_square(bbox):
    """Convert bbox to square

        Parameters:
        ----------
        bbox: numpy array , shape n x 5
            input bbox

        Returns:
        -------
        square bbox
        """
    square_bbox = bbox.copy()

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    max_side = np.maximum(h, w)
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
    return square_bbox
