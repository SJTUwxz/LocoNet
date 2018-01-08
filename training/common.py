#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: common.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/01/08
#   description:
#
#================================================================

import keras
import numpy as np
import tensorflow as tf


def preprocess_input(img):
    """preprocess input

    Args:
        img (TODO): TODO

    Returns: TODO

    """
    # swap rgb to bgr
    img = img[..., ::-1]
    # substract mean
    img -= [103.939, 116.779, 123.68]
    return img


def preprocess_output(num_classes):
    """preprocess output

    Args:
        label (TODO): TODO

    Returns: TODO

    """

    def preprocess(label):
        if isinstance(label, (np.ndarray, int, long)):
            label = keras.utils.to_categorical(label, num_classes)
        else:
            label = tf.one_hot(label, num_classes)
        return label

    return preprocess
