import numpy as np

from mtcnn import p_net, o_net, r_net
from .utils import load_weights, process_image, generate_bbox, py_nms


class Detector:
    def __init__(self, weight_dir,
                 slide_window=False,
                 stride=2,
                 min_face_size=24,
                 threshold=None,
                 scale_factor=0.79,
                 mode=3):

        assert mode in [1, 2, 3]
        assert scale_factor < 1

        self.slide_window = slide_window
        self.stride = stride
        self.min_face_size = min_face_size
        self.threshold = [0.6, 0.7, 0.7] if threshold is None else threshold
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

    def predict(self, image):
        if not self.slide_window:
            im_ = np.array(image)
            return self.predict_with_p_net(im_)
        else:
            raise NotImplementedError('Not implemented yet')

    def predict_with_p_net(self, im):

        current_scale = float(12) / self.min_face_size  # find initial scale
        im_resized = process_image(im, current_scale)
        current_height, current_width, _ = im_resized.shape

        all_boxes = []
        while min(current_height, current_width) > 12:
            inputs = np.array([im_resized])
            print('inputs shape: {}'.format(inputs.shape))
            labels, bboxes, landmarks = self.p_net.predict(inputs)
            labels = np.squeeze(labels)
            bboxes = np.squeeze(bboxes)

            boxes = generate_bbox(labels[:, :, 1], bboxes, current_scale, self.threshold[0])

            current_scale *= self.scale_factor
            im_resized = process_image(im, current_scale)
            current_height, current_width, _ = im_resized.shape

            if boxes.size == 0:
                continue

            keep = py_nms(boxes[:, :5], 0.7, 'union')
            boxes = boxes[keep]
            all_boxes.append(boxes)

        if len(all_boxes) == 0:
            return None, None, None

        return self.refine_bboxes(all_boxes)

    def predict_with_r_net(self, im, boxes):
        pass

    @staticmethod
    def refine_bboxes(all_boxes):
        all_boxes = np.vstack(all_boxes)
        # merge the detection from first stage
        keep = py_nms(all_boxes[:, 0:5], 0.5, 'union')
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
