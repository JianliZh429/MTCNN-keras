import os
import sys
from argparse import ArgumentParser

from train.data_loader import WiderFaceDataset
from train.train_net import train_p_net


def training(dataset_dir, batch_size, epochs, learning_rate):
    train_anno_file = os.path.join(dataset_dir, 'annotation_train.json')
    val_anno_file = os.path.join(dataset_dir, 'annotation_val.json')
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')

    train_dataset = WiderFaceDataset(train_dir, train_anno_file, batch_size, target_size=12)
    val_dataset = WiderFaceDataset(val_dir, val_anno_file, batch_size, target_size=12)



    train_p_net(train_x, train_y, val_x, val_y, batch_size, epochs, learning_rate)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset', type=str, help='Folder of training data')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size of training')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate while training')
    args = parser.parse_args(sys.argv[1:])

    training(args.dataset, args.batch_size, args.epochs, args.learning_rate)
