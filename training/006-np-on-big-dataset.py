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
from keras_retinanet.models.jh_resnet import ResNet50RetinaNet
from keras_retinanet.preprocessing.pascal_voc import PascalVocGenerator
from keras_retinanet.utils.keras_version import check_keras_version
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

from keras_extra.tfrecords_db import TfRecordDB
# from keras_extra.callbacks.tfrecord_eval import TfRecordEvalCallback
from keras_extra.callbacks import TfRecordEvalCallback


def create_model(
        weights='./data/snapshots/001-retinanet/resnet50_05-0.39389.h5',
        cls=2,
        fix_layers=False):
    image = keras.layers.Input((None, None, 3))
    model = ResNet50RetinaNet(image, num_classes=10, weights=weights)
    D7_pool = model.get_layer('D7_pool').output
    Global_cls = keras.layers.Dense(
        cls, activation='softmax', name='global_3cls')(D7_pool)
    new_model = keras.models.Model(inputs=model.inputs, outputs=[Global_cls])

    if fix_layers:
        for layer in new_model.layers:
            layer.trainable = False
        train_layers = ['D6', 'D7', 'D7_pool', 'global_3cls']
        for trainable_layer in train_layers:
            new_model.get_layer(trainable_layer).trainable = True
    return new_model


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

    def swap_rgb2bgr(img, label):
        img = img[..., ::-1]
        return img, label

    def substract_mean(img, label):
        img -= [103.939, 116.779, 123.68]
        return img, label

    def categorial_label(img, label):
        if isinstance(label, (np.ndarray, )):
            label = (label > 1).astype(int)
            label = keras.utils.to_categorical(label, num_classes)
        elif isinstance(label, (int, long)):
            label = int(label > 1)
            category = np.zeros(num_classes)
            category[label] = 1
            label = category
        else:
            label = tf.cast(tf.greater(label, 1), tf.int32)
            label = tf.one_hot(label, num_classes)
        return img, label

    post_processors = [swap_rgb2bgr, substract_mean, categorial_label]

    # train and val database
    train_db = TfRecordDB(train_label_file, 'train', train_record_path,
                          post_processors, num_classes, image_shape, channel)
    val_db = TfRecordDB(val_label_file, 'val', val_record_path,
                        post_processors, num_classes, image_shape, channel)

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
                     'snapshot_{epoch:02d}-{val_loss:.5f}--{val_acc:.5f}.h5'),
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

    class_weight = {0: 1, 1: 1.3}

    # Fit the model using data from the TFRecord data tensors.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    train_model.fit(
        epochs=50,
        steps_per_epoch=train_steps,
        callbacks=[evaluate_callback],
        class_weight=class_weight)
    # Clean up the TF session.
    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':

    import time
    import logging
    logging.getLogger().setLevel(logging.INFO)

    run_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    branch = '006-np-on-big-dataset'
    run_name = '{}/{}'.format(branch, run_time)
    logging.info('run_name: ' + run_name)

    num_classes = 2
    channel = 3

    batch_size = 32
    image_shape = (224, 224)
    train_record_path = './data/tf_records/full_nsp/train.record'
    val_record_path = './data/tf_records/full_nsp/val.record'
    # batch_size = 1
    # image_shape = (600, 1024)
    # train_record_path = './data/tf_records/10w_sp/train_w600-h1024.record'
    # val_record_path = './data/tf_records/10w_sp/val_w600-h1024.record'

    train_label_file = './data/labels/full-dataset/train.txt'
    val_label_file = './data/labels/full-dataset/val.txt'
    pre_trained_weight = './data/snapshots/001-retinanet/resnet50_05-0.39389.h5'
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
