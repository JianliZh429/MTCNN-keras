import os
import sys
from argparse import ArgumentParser

import numpy as np
from keras.optimizers import SGD, Adam

from mtcnn import p_net
from train.config import NET_SIZE
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
    _p_net.fit(inputs_image, y,
               batch_size=batch_size,
               initial_epoch=initial_epoch,
               epochs=epochs,
               callbacks=callbacks,
               verbose=1)
    return _p_net


def train_all_in_one(dataset_dir, batch_size, epochs, learning_rate, weights_file=None):
    label_dataset_path = os.path.join(dataset_dir, 'label_p_net.pkl')
    bboxes_dataset_path = os.path.join(dataset_dir, 'bboxes_p_net.pkl')
    landmarks_dataset_path = os.path.join(dataset_dir, 'landmarks_p_net.pkl')
    images_x, labels_y, bboxes_y, landmarks_y = load_dataset(label_dataset_path, bboxes_dataset_path,
                                                             landmarks_dataset_path, im_size=NET_SIZE['p_net'])

    callbacks, model_file = create_callbacks_model_file('p_net', epochs)
    _p_net = train_p(images_x, labels_y, bboxes_y, landmarks_y, batch_size, initial_epoch=0, epochs=epochs,
                     lr=learning_rate, callbacks=callbacks, weights_file=weights_file)

    _p_net.save_weights(model_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset', type=str, help='Folder of training data')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size of training')
    parser.add_argument('--epochs', type=int, default=30, help='Epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate while training')
    parser.add_argument('--weights', type=str, default=None, help='Init weights to load')
    args = parser.parse_args(sys.argv[1:])

    train_all_in_one(args.dataset, args.batch_size, args.epochs, args.learning_rate)
