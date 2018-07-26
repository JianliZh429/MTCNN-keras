import datetime
import os

from keras import Model, layers
from keras.callbacks import TensorBoard, ModelCheckpoint
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
                callbacks=None):
    _p_net = p_net(training=True)

    losses = {
        'p_classifier': 'binary_crossentropy',
        'p_bbox': 'mean_squared_error',
        'p_landmark': 'mean_squared_error'
    }
    # metrics = {
    #     'p_classifier': 'accuracy',
    #     'p_bbox': 'accuracy',
    #     'p_landmark': 'accuracy',
    #
    # }

    _p_net.compile(Adam(lr=lr), loss=losses, metrics=['accuracy'])
    _p_net.fit(
        inputs_image,
        {'p_classifier': labels, 'p_bbox': bboxes, 'p_landmark': landmarks},
        loss_weights={'p_classifier': 1., 'p_bbox': 0.5, 'p_landmark': 0.5},
        batch_size=batch_size,
        initial_epoch=initial_epoch,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    return _p_net

def train_p_net(p_net_, train_x, train_y, mode='label', batch_size=100, initial_epoch=0, epochs=1,
                learning_rate=0.001, callbacks=None):
    assert mode in MODES
    optimizer = Adam(lr=learning_rate)
    if mode == MODES[0]:
        classifier_layer = p_net_.get_layer('p_classifier')
        x = classifier_layer.output
        x = layers.Reshape((2,), name='p_classifier1')(x)

        model = Model(inputs=[p_net_.input], outputs=[x])
        model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    elif mode == MODES[1]:
        bbox_layer = p_net_.get_layer('p_bbox')
        x = bbox_layer.output
        x = layers.Reshape((4,), name='p_bbox1')(x)
        model = Model(inputs=[p_net_.input], outputs=[x])
        model.compile(optimizer, loss='mean_squared_error', metrics=['accuracy'])

    elif mode == MODES[2]:
        landmark_layer = p_net_.get_layer('p_landmark')
        x = landmark_layer.output
        x = layers.Reshape((10,), name='p_landmark1')(x)
        model = Model(inputs=[p_net_.input], outputs=[x])
        model.compile(optimizer, loss='mean_squared_error', metrics=['accuracy'])
    else:
        raise ValueError('Unknown mode of "{}", must in {}'.format(mode, MODES))

    # model.summary()

    model.fit(train_x, train_y,
              batch_size=batch_size,
              epochs=epochs,
              initial_epoch=initial_epoch,
              callbacks=callbacks,
              validation_split=0.1,
              verbose=1)
    return model, p_net_
