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


def create_model(
        weights='./data/snapshots/001-retinanet/resnet50_05-0.39389.h5',
        cls=3,
        fix_layers=False):
    image = keras.layers.Input((None, None, 3))
    model = ResNet50RetinaNet(
        image,
        num_classes=10,
        weights='./data/snapshots/001-retinanet/resnet50_05-0.39389.h5')
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


def train(run_name, train_label_file, val_label_file, batch_size=32):
    """train the model"""
    print('Creating model, this may take a second...')
    model = create_model()
    optimizer = keras.optimizers.sgd(
        lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
        loss={'global_3cls': 'categorical_crossentropy'},
        optimizer=optimizer,
        metrics=['accuracy'])
    # print model summary
    print(model.summary())

    image_data_generator = MyImageDataGenerator()
    train_gen = image_data_generator.flow_from_label_file(
        train_label_file,
        batch_size=batch_size,
        is_ergodic_files=None,
        balance=False)
    train_steps = train_gen.steps_per_epoch()
    val_gen = image_data_generator.flow_from_label_file(
        val_label_file,
        phase='val',
        batch_size=batch_size,
        is_ergodic_files=None,
        balance=False)
    val_steps = val_gen.steps_per_epoch()

    #callbacks
    snapshot_save_directory = './data/snapshots/{}'.format(run_name)
    log_save_directory = './data/logs/{}'.format(run_name)
    os.makedirs(snapshot_save_directory)
    os.makedirs(log_save_directory)
    model_check_point = keras.callbacks.ModelCheckpoint(
        os.path.join(snapshot_save_directory,
                     'snapshot_{epoch:02d}-{val_loss:.5f}--{val_acc:.5f}.h5'),
        monitor='val_loss',
        verbose=1,
        save_best_only=False)
    csv_logger = keras.callbacks.CSVLogger(
        os.path.join(log_save_directory, 'training_log.csv'))

    #train
    model.fit_generator(
        train_gen,
        steps_per_epoch=train_steps,
        epochs=50,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=[model_check_point, csv_logger])


if __name__ == '__main__':

    import time
    import logging
    logging.getLogger().setLevel(logging.INFO)

    run_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    branch = '002-finetune'
    run_name = '{}/{}'.format(branch, run_time)
    logging.info('run_name: ' + run_name)

    batch_size = 32
    train_label_file = './data/labels/10w_train.txt'
    val_label_file = './data/labels/10w_val.txt'

    train(
        batch_size=batch_size,
        train_label_file=train_label_file,
        val_label_file=val_label_file,
        run_name=run_name)
