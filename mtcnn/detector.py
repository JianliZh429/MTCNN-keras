import numpy as np

from mtcnn import p_net, o_net, r_net
from .utils import load_weights


class Detector:
    def __init__(self, weight_dir, slide_window=False, stride=2, mode=3):
        assert mode in [1, 2, 3]

        self.slide_window = slide_window
        self.stride = stride

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

    def predict(self, images):
        if not self.slide_window:
            im_batch = np.array(images)
            return self.predict_with_p_net(im_batch)
        else:
            raise NotImplementedError('Not implemented yet')

    def predict_with_p_net(self, im_batch):
        labels, bboxes, landmarks = self.p_net.predict(im_batch)
        return labels, bboxes, landmarks

    def _predict_with_slide_window(self, im_batch):
        pass
