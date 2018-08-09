import os
import sys
from argparse import ArgumentParser

import numpy as np
from keras.optimizers import Adam

from mtcnn import o_net
from train.config import NET_SIZE
from train.data_loader import load_dataset, DataGenerator
from train.train_helper import create_callbacks_model_file, loss_func

im_size = NET_SIZE['o_net']


def train_o_net(inputs_image, labels, bboxes, landmarks, batch_size, initial_epoch=0, epochs=1000, lr=0.001,
                callbacks=None, weights_file=None):
    y = np.concatenate((labels, bboxes, landmarks), axis=1)
    _o_net = o_net(training=True)
    _o_net.summary()
    if weights_file is not None:
        _o_net.load_weights(weights_file)

    _o_net.compile(Adam(lr=lr), loss=loss_func, metrics=['accuracy'])
    _o_net.fit(inputs_image, y,
               batch_size=batch_size,
               initial_epoch=initial_epoch,
               epochs=epochs,
               callbacks=callbacks,
               verbose=1)
    return _o_net


def train_o_net_with_data_generator(data_gen, steps_per_epoch, initial_epoch=0, epochs=1000, lr=0.001,
                                    callbacks=None, weights_file=None):
    _o_net = o_net(training=True)
    _o_net.summary()
    # optimizer = SGD(lr=lr, momentum=0.9, decay=0.01, nesterov=True)
    optimizer = Adam(lr=lr, decay=0.0001)

    if weights_file is not None:
        _o_net.load_weights(weights_file)

    _o_net.compile(optimizer, loss=loss_func, metrics=['accuracy'])

    _o_net.fit_generator(data_gen,
                         steps_per_epoch=steps_per_epoch,
                         initial_epoch=initial_epoch,
                         epochs=epochs,
                         callbacks=callbacks)

    return _o_net


def train_with_data_generator(dataset_dir, batch_size, epochs, learning_rate, weights_file=None):
    label_dataset_path = os.path.join(dataset_dir, 'label_o_net.h5')
    bboxes_dataset_path = os.path.join(dataset_dir, 'bboxes_o_net.h5')
    landmarks_dataset_path = os.path.join(dataset_dir, 'landmarks_o_net.h5')

    data_generator = DataGenerator(label_dataset_path, bboxes_dataset_path, landmarks_dataset_path, batch_size,
                                   im_size=NET_SIZE['o_net'], shuffle=True)
    data_gen = data_generator.generate()
    steps_per_epoch = data_generator.steps_per_epoch
    # data_generator.im_show(10)
    callbacks, model_file = create_callbacks_model_file('o_net', epochs)
    #
    _o_net = train_o_net_with_data_generator(data_gen, steps_per_epoch,
                                             initial_epoch=0,
                                             epochs=epochs,
                                             lr=learning_rate,
                                             callbacks=callbacks,
                                             weights_file=weights_file)
    _o_net.save_weights(model_file)


def train_all_in_one(dataset_dir, batch_size, epochs, learning_rate, weights_file=None):
    label_dataset_path = os.path.join(dataset_dir, 'label_o_net.h5')
    bboxes_dataset_path = os.path.join(dataset_dir, 'bboxes_o_net.h5')
    landmarks_dataset_path = os.path.join(dataset_dir, 'landmarks_o_net.h5')
    images_x, labels_y, bboxes_y, landmarks_y = load_dataset(label_dataset_path, bboxes_dataset_path,
                                                             landmarks_dataset_path, im_size=NET_SIZE['o_net'])

    callbacks, model_file = create_callbacks_model_file('o_net', epochs)

    _o_net = train_o_net(images_x, labels_y, bboxes_y, landmarks_y, batch_size, initial_epoch=0, epochs=epochs,
                         lr=learning_rate, callbacks=callbacks, weights_file=weights_file)

    _o_net.save_weights(model_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset', type=str, help='Folder of training data')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size of training')
    parser.add_argument('--epochs', type=int, default=1000, help='Epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate while training')
    parser.add_argument('--weights', type=str, default=None, help='Init weights to load')
    args = parser.parse_args(sys.argv[1:])

    train_with_data_generator(args.dataset, args.batch_size, args.epochs, args.learning_rate, args.weights)
