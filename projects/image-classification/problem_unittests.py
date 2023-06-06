import os
import numpy as np
import tensorflow as tf
import random
from unittest.mock import MagicMock


def _print_success_message():
    print('Tests Passed')


def test_folder_path(cifar10_dataset_folder_path):
    assert cifar10_dataset_folder_path is not None,\
        'Cifar-10 data folder not set.'
    assert cifar10_dataset_folder_path[-1] != '/',\
        'The "/" shouldn\'t be added to the end of the path.'
    assert os.path.exists(cifar10_dataset_folder_path),\
        'Path not found.'
    assert os.path.isdir(
        cifar10_dataset_folder_path
    ), f'{os.path.basename(cifar10_dataset_folder_path)} is not a folder.'

    train_files = [
        f'{cifar10_dataset_folder_path}/data_batch_{str(batch_id)}'
        for batch_id in range(1, 6)
    ]
    other_files = [
        f'{cifar10_dataset_folder_path}/batches.meta',
        f'{cifar10_dataset_folder_path}/test_batch',
    ]
    missing_files = [path for path in train_files + other_files if not os.path.exists(path)]

    assert not missing_files, f'Missing files in directory: {missing_files}'

    print('All files found!')


def test_normalize(normalize):
    test_shape = (np.random.choice(range(1000)), 32, 32, 3)
    test_numbers = np.random.choice(range(256), test_shape)
    normalize_out = normalize(test_numbers)

    assert type(normalize_out).__module__ == np.__name__,\
        'Not Numpy Object'

    assert (
        normalize_out.shape == test_shape
    ), f'Incorrect Shape. {normalize_out.shape} shape found'

    assert (
        normalize_out.max() <= 1 and normalize_out.min() >= 0
    ), f'Incorect Range. {normalize_out.min()} to {normalize_out.max()} found'

    _print_success_message()


def test_one_hot_encode(one_hot_encode):
    test_shape = np.random.choice(range(1000))
    test_numbers = np.random.choice(range(10), test_shape)
    one_hot_out = one_hot_encode(test_numbers)

    assert type(one_hot_out).__module__ == np.__name__,\
        'Not Numpy Object'

    assert one_hot_out.shape == (
        test_shape,
        10,
    ), f'Incorrect Shape. {one_hot_out.shape} shape found'

    n_encode_tests = 5
    test_pairs = list(zip(test_numbers, one_hot_out))
    test_indices = np.random.choice(len(test_numbers), n_encode_tests)
    labels = [test_pairs[test_i][0] for test_i in test_indices]
    enc_labels = np.array([test_pairs[test_i][1] for test_i in test_indices])
    new_enc_labels = one_hot_encode(labels)

    assert np.array_equal(
        enc_labels, new_enc_labels
    ), f'Encodings returned different results for the same numbers.\nFor the first call it returned:\n{enc_labels}\nFor the second call it returned\n{new_enc_labels}\nMake sure you save the map of labels to encodings outside of the function.'

    _print_success_message()


def test_nn_image_inputs(neural_net_image_input):
    image_shape = (32, 32, 3)
    nn_inputs_out_x = neural_net_image_input(image_shape)

    assert nn_inputs_out_x.get_shape().as_list() == [
        None,
        image_shape[0],
        image_shape[1],
        image_shape[2],
    ], f'Incorrect Image Shape.  Found {nn_inputs_out_x.get_shape().as_list()} shape'

    assert (
        nn_inputs_out_x.op.type == 'Placeholder'
    ), f'Incorrect Image Type.  Found {nn_inputs_out_x.op.type} type'

    assert (
        nn_inputs_out_x.name == 'x:0'
    ), f'Incorrect Name.  Found {nn_inputs_out_x.name}'

    print('Image Input Tests Passed.')


def test_nn_label_inputs(neural_net_label_input):
    n_classes = 10
    nn_inputs_out_y = neural_net_label_input(n_classes)

    assert nn_inputs_out_y.get_shape().as_list() == [
        None,
        n_classes,
    ], f'Incorrect Label Shape.  Found {nn_inputs_out_y.get_shape().as_list()} shape'

    assert (
        nn_inputs_out_y.op.type == 'Placeholder'
    ), f'Incorrect Label Type.  Found {nn_inputs_out_y.op.type} type'

    assert (
        nn_inputs_out_y.name == 'y:0'
    ), f'Incorrect Name.  Found {nn_inputs_out_y.name}'

    print('Label Input Tests Passed.')


def test_nn_keep_prob_inputs(neural_net_keep_prob_input):
    nn_inputs_out_k = neural_net_keep_prob_input()

    assert (
        nn_inputs_out_k.get_shape().ndims is None
    ), f'Too many dimensions found for keep prob.  Found {nn_inputs_out_k.get_shape().ndims} dimensions.  It should be a scalar (0-Dimension Tensor).'

    assert (
        nn_inputs_out_k.op.type == 'Placeholder'
    ), f'Incorrect keep prob Type.  Found {nn_inputs_out_k.op.type} type'

    assert (
        nn_inputs_out_k.name == 'keep_prob:0'
    ), f'Incorrect Name.  Found {nn_inputs_out_k.name}'

    print('Keep Prob Tests Passed.')


def test_con_pool(conv2d_maxpool):
    test_x = tf.placeholder(tf.float32, [None, 32, 32, 5])
    test_num_outputs = 10
    test_con_k = (2, 2)
    test_con_s = (4, 4)
    test_pool_k = (2, 2)
    test_pool_s = (2, 2)

    conv2d_maxpool_out = conv2d_maxpool(test_x, test_num_outputs, test_con_k, test_con_s, test_pool_k, test_pool_s)

    assert conv2d_maxpool_out.get_shape().as_list() == [
        None,
        4,
        4,
        10,
    ], f'Incorrect Shape.  Found {conv2d_maxpool_out.get_shape().as_list()} shape'

    _print_success_message()


def test_flatten(flatten):
    test_x = tf.placeholder(tf.float32, [None, 10, 30, 6])
    flat_out = flatten(test_x)

    assert flat_out.get_shape().as_list() == [
        None,
        10 * 30 * 6,
    ], f'Incorrect Shape.  Found {flat_out.get_shape().as_list()} shape'

    _print_success_message()


def test_fully_conn(fully_conn):
    test_x = tf.placeholder(tf.float32, [None, 128])
    test_num_outputs = 40

    fc_out = fully_conn(test_x, test_num_outputs)

    assert fc_out.get_shape().as_list() == [
        None,
        40,
    ], f'Incorrect Shape.  Found {fc_out.get_shape().as_list()} shape'

    _print_success_message()


def test_output(output):
    test_x = tf.placeholder(tf.float32, [None, 128])
    test_num_outputs = 40

    output_out = output(test_x, test_num_outputs)

    assert output_out.get_shape().as_list() == [
        None,
        40,
    ], f'Incorrect Shape.  Found {output_out.get_shape().as_list()} shape'

    _print_success_message()


def test_conv_net(conv_net):
    test_x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    test_k = tf.placeholder(tf.float32)

    logits_out = conv_net(test_x, test_k)

    assert logits_out.get_shape().as_list() == [
        None,
        10,
    ], f'Incorrect Model Output.  Found {logits_out.get_shape().as_list()}'

    print('Neural Network Built!')


def test_train_nn(train_neural_network):
    mock_session = tf.Session()
    test_x = np.random.rand(128, 32, 32, 3)
    test_y = np.random.rand(128, 10)
    test_k = np.random.rand(1)
    test_optimizer = tf.train.AdamOptimizer()

    mock_session.run = MagicMock()
    train_neural_network(mock_session, test_optimizer, test_k, test_x, test_y)

    assert mock_session.run.called, 'Session not used'

    _print_success_message()
