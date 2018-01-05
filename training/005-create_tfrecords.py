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


def create_tfrecord(train_label_file,
                    train_record_path,
                    val_label_file,
                    val_record_path,
                    num_classes,
                    image_shape=(224, 224),
                    channel=3):
    """train the model"""
    sess = keras.backend.get_session()

    post_processors = None
    train_db = TfRecordDB(train_label_file, 'train', train_record_path,
                          post_processors, num_classes, image_shape, channel)
    val_db = TfRecordDB(val_label_file, 'val', val_record_path,
                        post_processors, num_classes, image_shape, channel)

    val_db.write_record()


if __name__ == '__main__':

    import time
    import logging
    logging.getLogger().setLevel(logging.INFO)

    num_classes = 3
    image_shape = (224, 224)
    channel = 3
    train_record_path = './data/tf_records/full_nsp/train.record'
    val_record_path = './data/tf_records/full_nsp/val.record'

    train_label_file = './data/labels/full-dataset/train.txt'
    val_label_file = './data/labels/full-dataset/val.txt'
    create_tfrecord(train_label_file, train_record_path, val_label_file,
                    val_record_path, num_classes, image_shape, channel)
