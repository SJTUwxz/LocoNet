#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: tf_record_finetune.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   create date: 2017/12/29
#   description:
#
#================================================================

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.backend.tensorflow_backend import set_session
set_session(session)

import os
import sys
import keras
import numpy as np
import tensorflow as tf

from keras_extra.tfrecords_db import TfRecordDB


def create_tfrecord(label_file,
                    record_path,
                    prefix,
                    num_classes,
                    image_shape=(224, 224),
                    channel=3):
    """train the model"""
    sess = keras.backend.get_session()
    post_processors = None
    val_db = TfRecordDB(
        label_file=label_file,
        prefix=prefix,
        record_save_path=record_path,
        post_processors=post_processors,
        num_classes=num_classes,
        image_shape=image_shape,
        channel=channel)

    val_db.write_record()


if __name__ == '__main__':

    import time
    import logging
    logging.getLogger().setLevel(logging.INFO)

    num_classes = 3
    image_shape = (224, 224)
    channel = 3
    prefix = 'test'

    record_path = './data/tf_records/full_nsp/test.record'
    label_file = './data/labels/full-dataset/test.txt'
    create_tfrecord(label_file, record_path, prefix, num_classes, image_shape,
                    channel)
