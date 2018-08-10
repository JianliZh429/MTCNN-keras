import datetime
import os

import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint

LOG_DIR = os.path.join(os.path.dirname(__file__), '../logs')
MODES = ['label', 'bbox', 'landmark']

num_keep_radio = 0.7


def create_callbacks_model_file(prefix, epochs):
    filename = datetime.datetime.now().strftime('%Y%m%d_%H%M%S.%f')
    log_dir = "{}/{}_{}".format(LOG_DIR, prefix, filename)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    tensor_board = TensorBoard(log_dir=log_dir)
    model_file_path = '{}/{}_{}_{}.h5'.format(log_dir, prefix, epochs, filename)

    checkpoint = ModelCheckpoint(model_file_path, verbose=0, save_weights_only=True)
    return [checkpoint, tensor_board], model_file_path


def cal_mask(label_true, _type='label'):
    label_true_int32 = tf.cast(label_true, dtype=tf.int32)
    if _type == 'label':
        label_filtered = tf.map_fn(lambda x: tf.cond(tf.equal(x[0], x[1]), lambda: 0, lambda: 1), label_true_int32)
    elif _type == 'bbox':
        label_filtered = tf.map_fn(lambda x: tf.cond(tf.equal(x[0], 1), lambda: 0, lambda: 1), label_true_int32)
    elif _type == 'landmark':
        label_filtered = tf.map_fn(lambda x: tf.cond(tf.logical_and(tf.equal(x[0], 1), tf.equal(x[1], 1)),
                                                     lambda: 1, lambda: 0), label_true_int32)
    else:
        raise ValueError('Unknown type of: {} while calculate mask'.format(_type))

    mask = tf.cast(label_filtered, dtype=tf.int32)
    return mask


def _2_true_labels(label_true):
    """
    :param label_true: shape of (None, 2)
    :return: shape of (None, 1)
    None is batch size that can
    """
    return label_true[:, 0]


def label_ohem(label_true, label_pred):
    label = _2_true_labels(label_true)
    zeros = tf.zeros_like(label)
    label_filter_invalid = tf.where(tf.less(label, 0), zeros, label)

    label_int = tf.cast(label_filter_invalid, tf.int32)
    num_cls_prob = tf.size(label_pred)
    cls_prob_reshape = tf.reshape(label_pred, [num_cls_prob, -1])

    num_row = tf.to_int32(tf.shape(label_pred)[0])
    row = tf.range(num_row) * 2
    indices_ = row + label_int
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))
    loss = -tf.log(label_prob + 1e-10)
    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob, dtype=tf.float32)
    valid_inds = tf.where(label < zeros, zeros, ones)
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid * num_keep_radio, dtype=tf.int32)
    # set 0 to invalid sample
    loss = loss * valid_inds
    loss, _ = tf.nn.top_k(loss, k=keep_num)
    return tf.reduce_mean(loss)


def bbox_ohem(label_true, bbox_true, bbox_pred):
    label = _2_true_labels(label_true)

    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    ones_index = tf.ones_like(label, dtype=tf.float32)

    valid_inds = tf.where(tf.equal(tf.abs(label), 1), ones_index, zeros_index)

    square_error = tf.square(bbox_pred - bbox_true)
    square_error = tf.reduce_sum(square_error, axis=1)

    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    square_error = square_error * valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)

    return tf.reduce_mean(square_error)


def landmark_ohem(label_true, landmark_true, landmark_pred):
    label = _2_true_labels(label_true)

    ones = tf.ones_like(label, dtype=tf.float32)
    zeros = tf.zeros_like(label, dtype=tf.float32)
    valid_inds = tf.where(tf.equal(label, -2), ones, zeros)
    square_error = tf.square(landmark_pred - landmark_true)
    square_error = tf.reduce_sum(square_error, axis=1)
    num_valid = tf.reduce_sum(valid_inds)
    # keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    square_error = square_error * valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)


def loss_func(y_true, y_pred):
    labels_true = y_true[:, :2]
    bbox_true = y_true[:, 2:6]
    landmark_true = y_true[:, 6:]

    labels_pred = y_pred[:, :2]
    bbox_pred = y_pred[:, 2:6]
    landmark_pred = y_pred[:, 6:]

    label_loss = label_ohem(labels_true, labels_pred)
    bbox_loss = bbox_ohem(labels_true, bbox_true, bbox_pred)
    landmark_loss = landmark_ohem(labels_true, landmark_true, landmark_pred)

    return label_loss + bbox_loss * 0.5 + landmark_loss * 0.5


def metric_acc(y_true, y_pred):
    labels_true = y_true[:, :2]
    labels_pred = y_pred[:, :2]

    label = _2_true_labels(labels_true)

    pred = tf.argmax(labels_pred, axis=1)
    label_int = tf.cast(label, tf.int64)

    cond = tf.where(tf.greater_equal(label_int, 0))
    picked = tf.squeeze(cond)
    label_picked = tf.gather(label_int, picked)
    pred_picked = tf.gather(pred, picked)

    return tf.reduce_mean(tf.cast(tf.equal(label_picked, pred_picked), tf.float32))
