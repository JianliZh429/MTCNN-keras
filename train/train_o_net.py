import os
import sys
from argparse import ArgumentParser

from train.config import NET_SIZE
from train.data_loader import load_dataset, DataGenerator
from train.train_net import train_o_net, create_callbacks_model_file, train_o_net_with_data_generator

im_size = NET_SIZE['o_net']


def train_with_data_generator(dataset_dir, batch_size, epochs, learning_rate, weights_file=None):
    label_dataset_path = os.path.join(dataset_dir, 'label_o_net.h5')
    bboxes_dataset_path = os.path.join(dataset_dir, 'bboxes_o_net.h5')
    landmarks_dataset_path = os.path.join(dataset_dir, 'landmarks_o_net.h5')

    data_generator = DataGenerator(label_dataset_path, bboxes_dataset_path, landmarks_dataset_path, batch_size,
                                   im_size=NET_SIZE['o_net'])
    data_gen = data_generator.generate()
    steps_per_epoch = data_generator.steps_per_epoch()

    callbacks, model_file = create_callbacks_model_file('o_net', epochs)

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

    train_with_data_generator(args.dataset, args.batch_size, args.epochs, args.learning_rate)
