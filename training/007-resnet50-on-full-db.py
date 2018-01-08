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
import keras_retinanet.losses
from keras.applications import resnet50
from keras_retinanet.models.jh_resnet import ResNet50RetinaNet
from keras_retinanet.preprocessing.pascal_voc import PascalVocGenerator
from keras_retinanet.utils.keras_version import check_keras_version
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

from keras_extra.tfrecords_db import TfRecordDB
from keras_extra.callbacks.tfrecord_eval import TfRecordEvalCallback
from keras_extra.core.evaluate import evaluate


def create_model(weights='imagenet', cls=2, fix_layers=False):
    image = keras.layers.Input((224, 224, 3))
    base_model = resnet50.ResNet50(input_tensor=image, include_top=False)
    x = base_model.output
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(cls, activation='softmax', name='global_cls')(x)
    model = keras.models.Model(inputs=base_model.input, outputs=x)
    return model


def prepare_db(label_file, record_path, prefix, num_classes, image_shape,
               channel):
    """prepare record db
    Returns: TODO

    """

    def swap_rgb2bgr(img, label):
        img = img[..., ::-1]
        return img, label

    def substract_mean(img, label):
        img -= [103.939, 116.779, 123.68]
        return img, label

    def categorial_label(img, label):
        if isinstance(label, (np.ndarray, int, long)):
            label = keras.utils.to_categorical(label, num_classes)
        else:
            label = tf.one_hot(label, num_classes)
        return img, label

    post_processors = [swap_rgb2bgr, substract_mean, categorial_label]
    return TfRecordDB(label_file, prefix, record_path, post_processors,
                      num_classes, image_shape, channel)


def test(weights, label_file, record_path, batch_size, num_classes,
         image_shape, channel):
    """test on record db

    Args:
        model (TODO): TODO
        label_file (TODO): TODO
        record_path (TODO): TODO
        batch_size (TODO): TODO

    Returns: TODO

    """
    test_db = prepare_db(label_file, record_path, 'test', num_classes,
                         image_shape, channel)
    # create model
    model = create_model(cls=num_classes, fix_layers=False)
    model.load_weights(weights)
    result = evaluate(model, test_db, batch_size, verbose=1)
    print(
        'test_loss: {val_loss:.5f},test_acc: {val_acc:.5f}\nclassification repor:\n{class_report}\nconfusion maxtrix:\n{cm}\n'.
        format(**result))


def train(run_name,
          train_label_file,
          train_record_path,
          val_label_file,
          val_record_path,
          batch_size=32,
          num_classes=3,
          image_shape=(224, 224),
          channel=3,
          weights='./data/snapshots/001-retinanet/resnet50_05-0.39389.h5'):
    """train the model"""
    sess = keras.backend.get_session()

    # train and val database
    train_db = prepare_db(train_label_file, train_record_path, 'train',
                          num_classes, image_shape, channel)
    val_db = prepare_db(val_label_file, val_record_path, 'val', num_classes,
                        image_shape, channel)

    # steps per epoch
    train_steps = train_db.get_steps(batch_size)

    # create model
    model = create_model(cls=num_classes, fix_layers=False)

    # optimizer
    optimizer = keras.optimizers.sgd(
        lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    # optimizer = keras.optimizers.adam(lr=1e-5, clipnorm=0.001)

    #callbacks
    snapshot_save_directory = './data/snapshots/{}'.format(run_name)
    log_save_directory = './data/logs/{}'.format(run_name)
    os.makedirs(snapshot_save_directory)
    os.makedirs(log_save_directory)
    evaluate_callback = TfRecordEvalCallback(
        model, val_db,
        os.path.join(snapshot_save_directory,
                     'snapshot_{epoch:02d}-{val_loss:.5f}-{val_acc:.5f}.h5'),
        os.path.join(log_save_directory, 'training_log.csv'), batch_size)

    # build train model
    img_tensor, label_tensor = train_db.read_record(batch_size=batch_size)
    model_input = keras.layers.Input(tensor=img_tensor)
    train_model = model(model_input)
    train_model = keras.models.Model(inputs=model_input, outputs=train_model)

    # compile train model
    train_model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        target_tensors=[label_tensor],
        metrics=['accuracy'])

    # Fit the model using data from the TFRecord data tensors.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    train_model.fit(
        epochs=50, steps_per_epoch=train_steps, callbacks=[evaluate_callback])
    # Clean up the TF session.
    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':

    import time
    import logging
    logging.getLogger().setLevel(logging.INFO)

    run_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    branch = '007-resnet50-on-full-db'
    run_name = '{}/{}'.format(branch, run_time)
    logging.info('run_name: ' + run_name)

    num_classes = 3
    channel = 3

    batch_size = 32
    image_shape = (224, 224)
    train_record_path = './data/tf_records/full_nsp/train.record'
    val_record_path = './data/tf_records/full_nsp/val.record'
    test_record_path = './data/tf_records/full_nsp/test.record'

    train_label_file = './data/labels/full-dataset/train.txt'
    val_label_file = './data/labels/full-dataset/val.txt'
    test_label_file = './data/labels/full-dataset/test.txt'
    pre_trained_weight = './data/snapshots/001-retinanet/resnet50_05-0.39389.h5'
    weights = './data/snapshots/007-resnet50-on-full-db/20180107-023834/snapshot_00-0.13984--0.94756.h5'
    if sys.argv[1] == 'train':
        train(
            run_name,
            train_label_file,
            train_record_path,
            val_label_file,
            val_record_path,
            batch_size,
            num_classes,
            image_shape,
            channel,
            weights=pre_trained_weight)
    else:
        test(weights, test_label_file, test_record_path, batch_size,
             num_classes, image_shape, channel)
