import glob

import cv2
import numpy as np


def load_weights(weights_dir):
    weights_files = glob.glob('{}/*.h5'.format(weights_dir))
    p_net_weight = None
    r_net_weight = None
    o_net_weight = None
    for wf in weights_files:
        if 'p_net' in wf:
            p_net_weight = wf
        elif 'r_net' in wf:
            r_net_weight = wf
        elif 'o_net' in wf:
            o_net_weight = wf
        else:
            raise ValueError('No valid weights files found, should be p_net*.h5, r_net*.h5, o_net*.h5')

    if p_net_weight is None and r_net_weight is None and o_net_weight is None:
        raise ValueError('No valid weights files found, please specific the correct weights file directory')

    return p_net_weight, r_net_weight, o_net_weight


def process_image(img, scale):
    height, width, channels = img.shape
    new_height = int(height * scale)  # resized new height
    new_width = int(width * scale)  # resized new width
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)  # resized image
    img_resized = (img_resized - 127.5) / 128
    return img_resized


def batch_gen_bbox(cls_map, reg, scale, threshold, stride=2, cell_size=12):
    bboxes = []
    for cls, bbox in zip(cls_map, reg):
        b = generate_bbox(cls, bbox, scale, threshold, stride, cell_size)
        bboxes.append(b)
    return bboxes


def generate_bbox(cls_map, reg, scale, threshold, stride=2, cell_size=12):
    """
        generate bbox from feature cls_map
    Parameters:
    ----------
        cls_map: numpy array , n x m
            detect score for each position
        reg: numpy array , n x m x 4
            bbox
        scale: float number
            scale of this detection
        threshold: float number
            detect threshold
    Returns:
    -------
        bbox array
    """
    t_index = np.where(cls_map > threshold)

    # find nothing
    if t_index[0].size == 0:
        return np.array([])

    # offset
    dx1, dy1, dx2, dy2 = [reg[t_index[0], t_index[1], i] for i in range(4)]

    reg = np.array([dx1, dy1, dx2, dy2])
    score = cls_map[t_index[0], t_index[1]]
    bbox = np.vstack([np.round((stride * t_index[1]) / scale),
                      np.round((stride * t_index[0]) / scale),
                      np.round((stride * t_index[1] + cell_size) / scale),
                      np.round((stride * t_index[0] + cell_size) / scale),
                      score,
                      reg])

    return bbox.T


def py_nms(bboxes, thresh, mode="union"):
    """
    greedily select boxes with high confidence
    keep boxes overlap <= thresh
    rule out overlap > thresh
    :param mode:
    :param bboxes: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap <= thresh
    :return: indexes to keep
    """
    assert mode in ['union', 'minimum']

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    scores = bboxes[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        else:
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        # keep
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
