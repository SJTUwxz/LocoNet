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

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import resnet50
from keras_extra.generator import LabelFileIterator
from keras_extra.utils.image import preprocess_image


def create_model(weights='imagenet', cls=3, fix_layers=False):
    image = keras.layers.Input((None, None, 3))
    base_model = resnet50.ResNet50(
        input_tensor=image, include_top=False, weights=None)
    x = base_model.output
    # x = keras.layers.Flatten()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(cls, activation='softmax', name='global_3cls')(x)
    model = keras.models.Model(inputs=base_model.input, outputs=x)
    return model


def train(run_name,
          train_label_file,
          val_label_file,
          batch_size=32,
          num_classes=2,
          fix_layers=False):
    """train the model"""

    print('Creating model, this may take a second...')
    model = create_model(cls=num_classes, fix_layers=fix_layers)
    optimizer = keras.optimizers.sgd(
        lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
        loss={'global_3cls': 'categorical_crossentropy'},
        optimizer=optimizer,
        metrics=['accuracy'])
    # print model summary
    print(model.summary())

    train_image_data_generator = ImageDataGenerator()
    val_image_data_generator = ImageDataGenerator()

    def preprocess_func(img, label):
        """preprocess image and label

        Args:
            img (TODO): TODO
            label (TODO): TODO

        Returns: TODO

        """
        img = preprocess_image(img)
        # label -= 1
        return img, label

    train_gen = LabelFileIterator(
        label_file_path=train_label_file,
        image_data_generator=train_image_data_generator,
        batch_size=batch_size,
        num_classes=num_classes,
        preprocess_function=preprocess_func)
    train_steps = train_gen.samples // batch_size
    val_gen = LabelFileIterator(
        label_file_path=val_label_file,
        image_data_generator=val_image_data_generator,
        batch_size=batch_size,
        num_classes=num_classes,
        preprocess_function=preprocess_func)
    val_steps = val_gen.samples // batch_size

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
        # use_multiprocessing=True,
        # workers=4,
        callbacks=[model_check_point, csv_logger])


if __name__ == '__main__':

    import time
    import logging
    logging.getLogger().setLevel(logging.INFO)

    run_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    branch = '012-cls-resnet500-finetune'
    run_name = '{}/{}'.format(branch, run_time)
    logging.info('run_name: ' + run_name)

    num_classes = 3
    fix_layers = False
    batch_size = 1
    train_label_file = './data/labels/10w_train.txt'
    val_label_file = './data/labels/10w_val.txt'

    train(
        batch_size=batch_size,
        train_label_file=train_label_file,
        val_label_file=val_label_file,
        run_name=run_name,
        num_classes=num_classes,
        fix_layers=fix_layers)
