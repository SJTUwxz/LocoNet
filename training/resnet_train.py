#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: resnet_train.py
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
from keras_extra.applications.resnet50 import ResNet50
from training.common import preprocess_input, preprocess_output


def train(model,
          train_label_file,
          train_record_path,
          val_label_file,
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
        train_label_file,
        train_record_path,
        val_label_file,
        val_record_path,
        snapshot_save_path=snapshot_save_path,
        log_save_path=log_save_path,
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])

    model.fit(epochs=epochs)


def run(run_type,
        num_classes,
        input_shape,
        batch_size,
        preprocess_input_func,
        preprocess_output_func,
        weights,
        train_label_file,
        train_record_path,
        val_label_file,
        val_record_path,
        test_record_path,
        snapshot_save_path,
        log_save_path,
        epochs,
        predict_img_path=None):
    """TODO: Docstring for run.

    Args:
        run_type (TODO): TODO
        num_classes (TODO): TODO
        input_shape (TODO): TODO
        batch_size (TODO): TODO
        preprocess_input (TODO): TODO
        preprocess_output (TODO): TODO
        weights (TODO): TODO
        train_label_file (TODO): TODO
        train_record_path (TODO): TODO
        val_label_file (TODO): TODO
        val_record_path (TODO): TODO
        snapshot_save_path (TODO): TODO
        log_save_path (TODO): TODO
        epochs (TODO): TODO

    Kwargs:
        predict_img_path (TODO): TODO

    Returns: TODO

    """
    model = ResNet50(
        num_classes=num_classes,
        input_shape=input_shape,
        batch_size=batch_size,
        preprocess_input=preprocess_input_func,
        preprocess_output=preprocess_output_func,
        weights=weights)
    if run_type == 'eval':
        model.evaluate(test_record_path, 'test')
    elif run_type == 'train':
        train(
            model,
            train_label_file,
            train_record_path,
            val_label_file,
            val_record_path,
            snapshot_save_path,
            log_save_path,
            epochs=epochs)
    elif run_type == 'predict':
        path = predict_img_path
        result = model.predict(path)
        print('{} is predicted as : {}'.format(path, result))
