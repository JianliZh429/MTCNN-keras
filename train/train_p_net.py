import os
import sys
from argparse import ArgumentParser

from mtcnn import p_net
from train.data_loader import load_label_dataset, load_bbox_dataset, load_landmark_dataset, load_dataset
from train.train_net import train_p_net, train_p_net_, create_callbacks_model_file


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


def train_discretely(dataset_dir, batch_size, epochs, learning_rate, weights_file=None):
    label_dataset_path = os.path.join(dataset_dir, 'label_p_net.pkl')
    bboxes_dataset_path = os.path.join(dataset_dir, 'bboxes_p_net.pkl')
    landmarks_dataset_path = os.path.join(dataset_dir, 'landmarks_p_net.pkl')

    label_x, label_y = load_label_dataset(label_dataset_path)
    bbox_x, bbox_y = load_bbox_dataset(bboxes_dataset_path)
    landmark_x, landmark_y = load_landmark_dataset(landmarks_dataset_path)

    _p_net = p_net()
    if weights_file is not None:
        _p_net.load_weights(weights_file)

    label_weights = None
    bbox_weights = None
    landmark_weights = None
    callbacks, model_file = create_callbacks_model_file('p_net', epochs)

    sub_epochs_label = 1
    sub_epochs_bbox = 1
    sub_epochs_landmark = 1

    for i in range(epochs):
        start = i * (sub_epochs_label + sub_epochs_bbox + sub_epochs_landmark)

        end = start + sub_epochs_label
        model, _p_net = train_p_net(_p_net, label_x, label_y, 'label', batch_size, start, end, learning_rate, callbacks)
        label_classifier = model.get_layer('p_classifier')
        label_weights = label_classifier.get_weights()

        start = end
        end = start + sub_epochs_bbox
        model, _p_net = train_p_net(_p_net, bbox_x, bbox_y, 'bbox', batch_size, start, end, learning_rate, callbacks)
        bbox_layer = model.get_layer('p_bbox')
        bbox_weights = bbox_layer.get_weights()

        start = end
        end = start + sub_epochs_landmark
        model, _p_net = train_p_net(_p_net, landmark_x, landmark_y, 'landmark',
                                    batch_size, start, end, learning_rate, callbacks)
        landmark_layer = model.get_layer('p_landmark')
        landmark_weights = landmark_layer.get_weights()

    _p_net.summary()

    _p_net.get_layer('p_classifier').set_weights(label_weights)
    _p_net.get_layer('p_bbox').set_weights(bbox_weights)
    _p_net.get_layer('p_landmark').set_weights(landmark_weights)
    print('Save model to: {}'.format(model_file))
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
