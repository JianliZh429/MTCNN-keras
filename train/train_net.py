import datetime
import os

import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.losses import mean_squared_error
from keras.optimizers import Adam

from mtcnn import p_net

LOG_DIR = os.path.join(os.path.dirname(__file__), '../logs')
MODES = ['label', 'bbox', 'landmark']


def create_callbacks_model_file(prefix, epochs):
    filename = datetime.datetime.now().strftime('%Y%m%d_%H%M%S.%f')
    log_dir = "{}/{}_{}".format(LOG_DIR, prefix, filename)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    tensor_board = TensorBoard(log_dir=log_dir)
    model_file_path = '{}/{}_{}_{}.h5'.format(log_dir, prefix, epochs, filename)

    checkpoint = ModelCheckpoint(model_file_path, verbose=0, save_weights_only=True)
    return [checkpoint, tensor_board], model_file_path


def train_p_net_(inputs_image, labels, bboxes, landmarks, batch_size, initial_epoch=0, epochs=1000, lr=0.001,
                 callbacks=None, weights_file=None):
    def loss_func(y_true, y_pred):

        return mean_squared_error(y_true, y_pred)

    y = np.concatenate((labels, bboxes, landmarks), axis=1)
    _p_net = p_net(training=True)
    _p_net.summary()
    if weights_file is not None:
        _p_net.load_weights(weights_file)

    _p_net.compile(Adam(lr=lr), loss=loss_func, metrics=['accuracy'])
    _p_net.fit(inputs_image, y,
               batch_size=batch_size,
               initial_epoch=initial_epoch,
               epochs=epochs,
               callbacks=callbacks,
               verbose=1)
    return _p_net
