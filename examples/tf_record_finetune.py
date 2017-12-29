#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.backend.tensorflow_backend import set_session
set_session(session)

import os
import keras
import keras_retinanet.losses
from keras_retinanet.models.jh_resnet import ResNet50RetinaNet
from keras_retinanet.preprocessing.pascal_voc import PascalVocGenerator
from keras_retinanet.utils.keras_version import check_keras_version
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

from preprocess.MyImageGenerator import MyImageDataGenerator
from preprocess.tf_record_db import TfRecordDb
from preprocess.evaluate_tf_record_callback import EvaluateTfRecordCallback


def create_model(
        weights='./data/snapshots/001-retinanet/resnet50_05-0.39389.h5',
        cls=3,
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
          train_record_path,
          val_record_path,
          batch_size=32,
          weights='./data/snapshots/001-retinanet/resnet50_05-0.39389.h5'):
    """train the model"""
    sess = keras.backend.get_session()

    num_train = 103780
    train_steps = num_train // batch_size

    # build train model
    model = create_model(fix_layers=False)
    train_gen = TfRecordDb(None, 'train', train_record_path)
    images, labels = train_gen.read_record(batch_size=batch_size)
    model_input = keras.layers.Input(tensor=images)
    train_model = model(model_input)
    train_model = keras.models.Model(inputs=model_input, outputs=train_model)
    optimizer = keras.optimizers.sgd(
        lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    train_model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        target_tensors=[labels],
        metrics=['accuracy'])

    # print model summary
    # print(model.summary())
    # print(train_model.summary())

    #callbacks
    val_gen = TfRecordDb(None, 'val', val_record_path)
    snapshot_save_directory = './data/snapshots/{}'.format(run_name)
    log_save_directory = './data/logs/{}'.format(run_name)
    os.makedirs(snapshot_save_directory)
    os.makedirs(log_save_directory)
    evaluate_callback = EvaluateTfRecordCallback(
        model, val_gen, sess,
        os.path.join(snapshot_save_directory,
                     'snapshot_{epoch:02d}-{val_loss:.5f}--{val_acc:.5f}.h5'),
        os.path.join(log_save_directory, 'training_log.csv'))

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
    branch = 'integrate-tfrecord'
    run_name = '{}/{}'.format(branch, run_time)
    logging.info('run_name: ' + run_name)

    batch_size = 32
    train_record_path = './data/tf_records/10w_nsp/train.record'
    val_record_path = './data/tf_records/10w_nsp/val.record'

    train(
        batch_size=batch_size,
        train_record_path=train_record_path,
        val_record_path=val_record_path,
        run_name=run_name)
