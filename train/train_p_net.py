import os
import sys
from argparse import ArgumentParser

import numpy as np
from keras import Model

from train.data_loader import load_dataset
from train.train_net import train_p_net, create_callbacks_model_file


def training(dataset_dir, batch_size, epochs, learning_rate):
    label_dataset_path = os.path.join(dataset_dir, 'label_p_net.pkl')
    bboxes_dataset_path = os.path.join(dataset_dir, 'bboxes_p_net.pkl')
    label_dataset = load_dataset(label_dataset_path)
    bbox_dataset = load_dataset(bboxes_dataset_path)

    label_x = label_dataset['ims']
    label_y = label_dataset['labels']
    label_x = np.array(label_x)
    label_y = np.array(label_y)

    bbox_x = bbox_dataset['ims']
    bbox_y = bbox_dataset['bboxes']
    bbox_x = np.array(bbox_x)
    bbox_y = np.array(bbox_y)

    _p_net = None
    label_weights = None
    bbox_weights = None
    callbacks, model_file = create_callbacks_model_file('p_net', epochs)
    for i in range(epochs):
        start = i * 4
        end = start + 1
        model, _p_net = train_p_net(label_x, label_y, 'label', batch_size, start, end, learning_rate, callbacks)
        label_classifier = model.get_layer('p_classifier')
        label_weights = label_classifier.get_weights()

        start = end
        end = start + 3
        model, _p_net = train_p_net(bbox_x, bbox_y, 'bbox', batch_size, start, end, learning_rate, callbacks)
        bbox_layer = model.get_layer('p_bbox')
        bbox_weights = bbox_layer.get_weights()

    model = Model([_p_net.input], [_p_net.get_layer('p_classifier').output, _p_net.get_layer('p_bbox').output])
    model.summary()

    model.get_layer('p_classifier').set_weights(label_weights)
    model.get_layer('p_bbox').set_weights(bbox_weights)

    model.save_weights(model_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset', type=str, help='Folder of training data')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size of training')
    parser.add_argument('--epochs', type=int, default=1000, help='Epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate while training')
    args = parser.parse_args(sys.argv[1:])

    training(args.dataset, args.batch_size, args.epochs, args.learning_rate)
