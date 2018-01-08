#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: 107-resnet50-on-full-db.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/01/08
#   description:
#
#================================================================

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.backend.tensorflow_backend import set_session
set_session(session)

import keras
import numpy as np
from keras_extra.applications.resnet50 import ResNet50

from training.common import preprocess_input, preprocess_output


def train(model,
          train_record_path,
          val_record_path,
          snapshot_save_path,
          log_save_path,
          epochs=50):
    """TODO: Docstring for train.
    Returns: TODO

    """
    # optimizer
    optimizer = keras.optimizers.sgd(
        lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)

    # compile train model
    model.compile(
        snapshot_save_path=snapshot_save_path,
        log_save_path=log_save_path,
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])

    model.fit(
        train_record_path,
        val_record_path,
        epochs=epochs,
        initial_epoch=initial_epoch)


if __name__ == "__main__":
    import sys
    import time
    import logging
    logging.getLogger().setLevel(logging.INFO)
    run_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    branch = '107-resnet50-on-full-db'
    run_name = '{}/{}'.format(branch, run_time)
    logging.info('run_name: ' + run_name)

    train_record_path = './data/tf_records/full_nsp/train.record'
    val_record_path = './data/tf_records/full_nsp/val.record'
    test_record_path = './data/tf_records/full_nsp/test.record'
    num_classes = 3
    input_shape = (224, 224, 3)
    batch_size = 32
    preprocess_input_func = preprocess_input
    preprocess_output_func = preprocess_output(num_classes)
    snapshot_save_path = './data/snapshots/{}'.format(
        run_name) + 'snapshot_{epoch:02d}-{val_loss:.5f}-{val_acc:.5f}.h5'
    log_save_path = './data/logs/{}'.format(run_name) + 'training_log.csv'
    weights = './data/snapshots/007-resnet50-on-full-db/20180107-023834/snapshot_00-0.13984--0.94756.h5'

    model = ResNet50(
        num_classes=num_classes,
        input_shape=input_shape,
        batch_size=batch_size,
        preprocess_input=preprocess_input_func,
        preprocess_output=preprocess_output_func,
        weights=weights)
    if sys.argv[1] == 'eval':
        model.evaluate(test_record_path, 'test')
    elif sys.argv[1] == 'train':
        train(
            model,
            train_record_path,
            val_record_path,
            snapshot_save_path,
            log_save_path,
            epochs=50)
    elif sys.argv[1] == 'predict':
        path = sys.argv[2]
        result = model.predict(path)
        print('{} is predicted as : {}'.format(path, result))
