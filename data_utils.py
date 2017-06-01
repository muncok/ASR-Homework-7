import numpy as np
import os
import tensorflow as tf



def dense_to_sparse(dense):
    with tf.Session() as sess:
        dense_t = tf.constant(dense)
        idx = tf.where(tf.not_equal(dense_t, 0))
        # Use tf.shape(a_t, out_type=tf.int64) instead of a_t.get_shape() if tensor shape is dynamic
        sparse = tf.SparseTensor(idx, tf.gather_nd(dense_t, idx), dense_t.get_shape())
        rdense_t = tf.sparse_tensor_to_dense(sparse)
        rdense = sess.run(rdense_t)
    assert (np.all(dense == rdense))


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def ctc_label(labels, blank_idx=30):
    ctc_labels = []
    for label in labels:
        ctc_label = []
        for char in label:
            ctc_label.append(blank_idx)
            ctc_label.append(char)
        ctc_label.append(blank_idx)
        ctc_labels.append(ctc_label)
    return ctc_labels


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok]*max_length_word,
                                            max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)


    return sequence_padded, sequence_length


def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)
    Returns:
        list of tuples
    """
    from sklearn.utils import shuffle
    shuffled_data = shuffle(data)
    x_batch, y_batch = [], []
    for (x, y) in shuffled_data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch


def normalize_mfcc(mfcc):
    """
    ref: https://github.com/muncok/timit_tf/blob/master/src/timittf/preprocessing.py 
    Normalize mfcc data using the following formula:

    normalized = (mfcc - mean)/standard deviation

    Args:
        mfcc (numpy.ndarray):
            An ndarray containing mfcc data.
            Its shape is [samples, sentence_length, coefficients]

    Returns:
        numpy.ndarray:
            An ndarray containing normalized mfcc data with the same shape as
            the input.
    """

    means = np.mean(mfcc, 0)
    stds = np.std(mfcc, 0)
    return (mfcc - means) / stds
