import datetime
import os

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam

from mtcnn import p_net

LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
P_NET = p_net()
MODES = ['label', 'bbox', 'landmark']


def create_callbacks_model_file(prefix, epochs):
    filename = datetime.datetime.now().strftime('%Y%m%d_%H%M%S.%f')
    log_dir = "{}/{}_{}".format(LOG_DIR, prefix, filename)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    tensor_board = TensorBoard(log_dir=log_dir)
    model_file_path = '{}/{}_{}_{}.h5'.format(log_dir, prefix, epochs, filename)

    checkpoint = ModelCheckpoint(model_file_path, verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min', period=1)
    return [checkpoint, tensor_board], model_file_path


def train_p_net(train_x, train_y, mode='label', batch_size=100, epochs=1000, learning_rate=0.001):
    assert mode in MODES
    losses = {'p_classifier': 'binary_crossentropy'}
    metrics = {'p_classifier': 'accuracy'}
    train_y_true = {'p_classifier': train_y[0]}
    if mode == MODES[1]:
        losses['p_bbox'] = 'mean_squared_error'
        metrics['p_bbox'] = 'accuracy'
        train_y_true['p_bbox'] = train_y[1]

    if mode == MODES[2]:
        losses['p_bbox'] = 'mean_squared_error'
        metrics['p_bbox'] = 'accuracy'
        train_y_true['p_bbox'] = train_y[1]

        losses['p_landmark'] = 'mean_squared_error'
        metrics['p_landmark'] = 'accuracy'
        train_y_true['p_landmark'] = train_y[2]

    callback_list, model_file = create_callbacks_model_file('p_net', epochs)

    P_NET.compile(optimizer=Adam(lr=learning_rate), loss=losses, metrics=metrics)
    P_NET.fit(train_x, train_y_true, batch_size=batch_size, epochs=epochs, callbacks=callback_list, verbose=1)
    P_NET.save(model_file)
