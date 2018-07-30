import os
import sys
from argparse import ArgumentParser

from train.data_loader import load_dataset
from train.train_net import train_p_net_, create_callbacks_model_file


def train_all_in_one(dataset_dir, batch_size, epochs, learning_rate, weights_file=None):
    label_dataset_path = os.path.join(dataset_dir, 'label_p_net.pkl')
    bboxes_dataset_path = os.path.join(dataset_dir, 'bboxes_p_net.pkl')
    landmarks_dataset_path = os.path.join(dataset_dir, 'landmarks_p_net.pkl')
    images_x, labels_y, bboxes_y, landmarks_y = load_dataset(label_dataset_path, bboxes_dataset_path,
                                                             landmarks_dataset_path, im_size=12)

    callbacks, model_file = create_callbacks_model_file('p_net', epochs)
    _p_net = train_p_net_(images_x, labels_y, bboxes_y, landmarks_y, batch_size, initial_epoch=0, epochs=epochs,
                          lr=learning_rate, callbacks=callbacks, weights_file=weights_file)

    _p_net.save_weights(model_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset', type=str, help='Folder of training data')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size of training')
    parser.add_argument('--epochs', type=int, default=1000, help='Epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate while training')
    parser.add_argument('--weights', type=str, default=None, help='Init weights to load')
    args = parser.parse_args(sys.argv[1:])

    train_all_in_one(args.dataset, args.batch_size, args.epochs, args.learning_rate)
