import datetime
import os

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam

from mtcnn import p_net
from .config import LOG_DIR

P_NET = p_net()


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


def train_p_net(train_x, train_y, val_x, val_y, batch_size=100, epochs=1000, learning_rate=0.001):
    # def loss_face_classifer(y_true, y_pred):
    #     p = np.log(y_pred)
    #     loss = -(y_true * p + (1 - y_true) * (1 - p))
    #     mean_squared_error
    #     return loss
    #
    # def loss_bbox(y_true, y_pred):
    #     return np.linalg.norm(y_true - y_pred)
    #
    # def loss_landmark(y_true, y_pred):
    losses = {
        'p_classifier': 'binary_crossentropy',
        'p_bbox': 'mean_squared_error',
        'p_landmark': 'mean_squared_error'
    }
    metricses = {
        'p_classifier': 'accuracy',
        'p_bbox': 'accuracy',
        'p_landmark': 'accuracy'
    }
    P_NET.compile(optimizer=Adam(lr=learning_rate), loss=losses, metrics=metricses)
    train_y_true = {
        'p_classifier': train_y[0],
        'p_bbox': train_y[1],
        'p_landmark': train_y[2]
    }
    val_y_true = {
        'p_classifier': val_y[0],
        'p_bbox': val_y[1],
        'p_landmark': val_y[2]
    }
    callback_list, model_file = create_callbacks_model_file('p_net', epochs)
    P_NET.fit(train_x, train_y_true,
              validation_data=(val_x, val_y_true),
              batch_size=batch_size,
              epochs=epochs,
              callbacks=callback_list,
              verbose=1)
    P_NET.save(model_file)
