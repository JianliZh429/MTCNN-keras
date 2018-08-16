import cv2
import numpy as np
from progress.bar import Bar

from mtcnn import p_net, o_net, r_net
from .utils import load_weights, process_image, generate_bbox, py_nms, py_nms2, bbox_2_square, pad, calibrate_bbox


class Detector:
    def __init__(self, weight_dir,
                 slide_window=False,
                 stride=2,
                 min_face_size=24,
                 threshold=None,
                 scale_factor=0.7,
                 mode=3):

        assert mode in [1, 2, 3]
        assert scale_factor < 1

        self.slide_window = slide_window
        self.stride = stride
        self.min_face_size = min_face_size
        self.threshold = [0.9, 0.7, 0.7] if threshold is None else threshold
        self.scale_factor = scale_factor

        self.p_net = None
        self.r_net = None
        self.o_net = None
        self.init_network(mode, weight_dir)

    def init_network(self, mode, weight_dir):
        p_weights, r_weights, o_weights = load_weights(weight_dir)
        print('PNet weight file is: {}'.format(p_weights))
        self.p_net = p_net()
        self.p_net.load_weights(p_weights)
        if mode > 1:
            self.r_net = r_net()
            self.r_net.load_weights(r_weights)
        if mode > 2:
            self.o_net = o_net()
            self.o_net.load_weights(o_weights)

    def predict(self, np_images, verbose=False):

        if not self.slide_window:
            all_boxes = []  # save each image's bboxes
            landmarks = []
            bar = None
            if verbose:
                bar = Bar('Detecting...', max=len(np_images))
            for im in np_images:
                if bar:
                    bar.next()
                boxes, boxes_c, landmark = self.predict_with_p_net(im)
                if boxes_c is None:
                    print("boxes_c is None...")
                    all_boxes.append(np.array([]))
                    landmarks.append(np.array([]))
                    continue
                if self.r_net is not None:
                    boxes, boxes_c, landmark = self.predict_with_r_net(im, boxes_c)
                    if boxes_c is None:
                        print("boxes_c is None...")
                        all_boxes.append(np.array([]))
                        landmarks.append(np.array([]))
                        continue
                if self.o_net is not None:
                    boxes, boxes_c, landmark = self.predict_with_o_net(im, boxes_c)
                    if boxes_c is None:
                        print("boxes_c is None...")
                        all_boxes.append(np.array([]))
                        landmarks.append(np.array([]))
                        continue

                all_boxes.append(boxes_c)
                landmarks.append(landmark)
            if bar:
                bar.finish()
            return all_boxes, landmarks
        else:
            raise NotImplementedError('Not implemented yet')

    def predict_with_p_net(self, im):

        current_scale = float(12) / self.min_face_size  # find initial scale
        im_resized = process_image(im, current_scale)
        current_height, current_width, _ = im_resized.shape

        all_boxes = []
        while min(current_height, current_width) > 12:
            inputs = np.array([im_resized])
            labels, bboxes, landmarks = self.p_net.predict(inputs)
            labels = labels[0]
            bboxes = bboxes[0]

            boxes = generate_bbox(labels[:, :, 1], bboxes, current_scale, self.threshold[0])

            current_scale *= self.scale_factor
            im_resized = process_image(im, current_scale)
            current_height, current_width, _ = im_resized.shape

            if boxes.size == 0:
                continue

            # keep = py_nms(boxes[:, :5], 0.1, 'union')
            keep = py_nms2(boxes[:, :5], 0.1)
            boxes = boxes[keep]
            all_boxes.append(boxes)

        if len(all_boxes) == 0:
            return None, None, None

        return self.refine_bboxes(all_boxes)

    def predict_with_r_net(self, im, boxes):
        h, w, c = im.shape
        box = bbox_2_square(boxes)
        box[:, 0:4] = np.round(box[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmp_w, tmp_h] = pad(box, w, h)
        num_boxes = box.shape[0]
        cropped_ims = np.zeros((num_boxes, 24, 24, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmp_h[i], tmp_w[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (24, 24)) - 127.5) / 128

        cls_scores, reg, _ = self.r_net.predict(cropped_ims)
        cls_scores = cls_scores[:, 1]
        keep_indices = np.where(cls_scores > self.threshold[1])[0]
        if len(keep_indices) > 0:
            boxes = box[keep_indices]
            boxes[:, 4] = cls_scores[keep_indices]
            reg = reg[keep_indices]
            # landmark = landmark[keep_inds]
        else:
            return None, None, None

        keep = py_nms(boxes, 0.6)
        boxes = boxes[keep]
        boxes_c = calibrate_bbox(boxes, reg[keep])
        return boxes, boxes_c, None

    def predict_with_o_net(self, im, bbox):
        """Get face candidates using onet

        Parameters:
        ----------
        im: numpy array
            input image array
        bbox: numpy array
            detection results of rnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """

        h, w, c = im.shape
        box = bbox_2_square(bbox)
        box[:, 0:4] = np.round(box[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmp_w, tmp_h] = pad(box, w, h)
        num_boxes = box.shape[0]
        cropped_ims = np.zeros((num_boxes, 48, 48, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmp_h[i], tmp_w[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (48, 48)) - 127.5) / 128

        cls_scores, reg, landmark = self.o_net.predict(cropped_ims)
        # prob belongs to face
        cls_scores = cls_scores[:, 1]
        keep_indices = np.where(cls_scores > self.threshold[2])[0]
        if len(keep_indices) > 0:
            # pick out filtered box
            boxes = box[keep_indices]
            boxes[:, 4] = cls_scores[keep_indices]
            reg = reg[keep_indices]
            landmark = landmark[keep_indices]
        else:
            return None, None, None

        # width
        w = boxes[:, 2] - boxes[:, 0] + 1
        # height
        h = boxes[:, 3] - boxes[:, 1] + 1
        landmark[:, 0::2] = (np.tile(w, (5, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (5, 1)) - 1).T
        landmark[:, 1::2] = (np.tile(h, (5, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (5, 1)) - 1).T
        boxes_c = calibrate_bbox(boxes, reg)

        boxes = boxes[py_nms(boxes, 0.6, "minimum")]
        keep = py_nms(boxes_c, 0.6, "minimum")
        boxes_c = boxes_c[keep]
        landmark = landmark[keep]
        return boxes, boxes_c, landmark

    @staticmethod
    def refine_bboxes(all_boxes):
        all_boxes = np.vstack(all_boxes)
        # merge the detection from first stage
        keep = py_nms(all_boxes[:, 0:5], 0.3, 'union')
        # keep = py_nms2(all_boxes[:, :5], 0.7)
        all_boxes = all_boxes[keep]
        boxes = all_boxes[:, :5]
        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
        # refine the boxes
        boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                             all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                             all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                             all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                             all_boxes[:, 4]])
        boxes_c = boxes_c.T
        return boxes, boxes_c, None

    def _predict_with_slide_window(self, im_batch):
        pass
