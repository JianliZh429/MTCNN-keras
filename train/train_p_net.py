import os
import random
import sys
from argparse import ArgumentParser

import numpy as np
from keras.optimizers import Adam, SGD

from mtcnn import p_net
from train.constants import NET_SIZE
from train.data_loader import load_dataset
from train.train_helper import create_callbacks_model_file, loss_func, metric_acc


def train_p(inputs_image, labels, bboxes, landmarks, batch_size, initial_epoch=0, epochs=1000, lr=0.001,
            callbacks=None, weights_file=None):
    y = np.concatenate((labels, bboxes, landmarks), axis=1)
    _p_net = p_net(training=True)
    _p_net.summary()
    if weights_file is not None:
        _p_net.load_weights(weights_file)
    optimizer = Adam(lr=lr)
    # optimizer = SGD(lr, momentum=0.9, decay=0.0001, nesterov=True)
    _p_net.compile(optimizer, loss=loss_func, metrics=[metric_acc])

    steps_per_epoch = len(inputs_image) / batch_size
    _p_net.fit(inputs_image, y,
               batch_size=batch_size,
               steps_per_epoch=steps_per_epoch,
               initial_epoch=initial_epoch,
               epochs=epochs,
               callbacks=callbacks,
               verbose=1)
    return _p_net


def train_all_in_one(dataset_dir, batch_size, epochs, learning_rate, weights_file=None):
    label_dataset_path = os.path.join(dataset_dir, 'label_p_net.h5')
    bboxes_dataset_path = os.path.join(dataset_dir, 'bboxes_p_net.h5')
    landmarks_dataset_path = os.path.join(dataset_dir, 'landmarks_p_net.h5')
    images_x, labels_y, bboxes_y, landmarks_y = load_dataset(label_dataset_path, bboxes_dataset_path,
                                                             landmarks_dataset_path, im_size=NET_SIZE['p_net'])

    callbacks, model_file = create_callbacks_model_file('p_net', epochs)
    _p_net = train_with_generator(images_x, labels_y, bboxes_y, landmarks_y, batch_size, epochs=epochs,
                                  lr=learning_rate, callbacks=callbacks, weights_file=weights_file)

    _p_net.save_weights(model_file)


def train_with_generator(inputs_image, labels, bboxes, landmarks, batch_size, epochs=1000, lr=0.001,
                         callbacks=None, weights_file=None):
    y = np.concatenate((labels, bboxes, landmarks), axis=1)
    _p_net = p_net(training=True)
    _p_net.summary()
    if weights_file is not None:
        _p_net.load_weights(weights_file)
    optimizer = Adam(lr=lr)
    # optimizer = SGD(lr=lr, momentum=0.9, decay=lr / 10., nesterov=True)
    _p_net.compile(optimizer, loss=loss_func, metrics=[metric_acc])

    length = len(inputs_image)

    def gen():
        while True:
            indices = random.sample(range(0, length), batch_size)
            x_train = []
            y_train = []
            for i in indices:
                x_train.append(inputs_image[i])
                y_train.append(y[i])
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            # print('::::::::', x_train.shape)
            # print('::::::::', y_train.shape)
            yield x_train, y_train

    steps_per_epoch = len(inputs_image) / batch_size

    train_gen = gen()

    _p_net.fit_generator(train_gen, steps_per_epoch, epochs=epochs, verbose=1, callbacks=callbacks)

    return _p_net


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset', type=str, help='Folder of training data')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size of training')
    parser.add_argument('--epochs', type=int, default=30, help='Epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate while training')
    parser.add_argument('--weights', type=str, default=None, help='Init weights to load')
    args = parser.parse_args(sys.argv[1:])

    train_all_in_one(args.dataset, args.batch_size, args.epochs, args.lr)
