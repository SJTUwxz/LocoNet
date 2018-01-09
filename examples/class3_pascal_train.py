#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This file trains all three classes, including unannotated normal images

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)

import keras
from keras.callbacks import CSVLogger
import argparse
import keras_retinanet.losses
from keras_retinanet.models.jh_resnet import ResNet50RetinaNet
from keras_retinanet.preprocessing.pascal_voc import PascalVocGenerator
from keras_retinanet.utils.keras_version import check_keras_version
from keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import SGD
# from preprocessing.image import ImageDataGenerator

import keras.preprocessing.image

#from preprocess.MyImageGenerator import MyImageDataGenerator


def get_session(gpu_fraction=0.7):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
    # config.gpu_options.allow_growth = true
    return tf.Session(config=config)

def create_model(weights='imagenet'):
    image = keras.layers.Input((None, None, 3))
    return ResNet50RetinaNet(
        image,
        num_classes=10,
        weights=
        '/home/xiziwang/projects/retinanet/data/snapshots/000-retinanet50-on-pascal/resnet50_05-0.39389.h5')


def parse_args():
    parser = argparse.ArgumentParser(description='Simple training script for Pascal VOC object detection.')
    parser.add_argument('voc_path', help='Path to Pascal VOC directory (ie. /tmp/VOCdevkit/VOC2007).')
    parser.add_argument('--weights', help='Weights to use for initialization (defaults to ImageNet).', default='imagenet')
    parser.add_argument('--batch-size', help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')

    return parser.parse_args()


if __name__ == '__main__':
    set_session(session)

    args = parse_args()

    check_keras_version()
    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        keras.backend.tensorflow_backend.set_session(get_session())

    # create the model
    print('Creating model, this may take a second...')

    model = create_model()

    csv_logger = CSVLogger('./data/logs/class3_pascal_train.log')

    D7_pool = model.get_layer('D7_pool').output
    Global_cls = keras.layers.Dense(
        3, activation='softmax', name='global_3cls')(D7_pool)

    model.outputs[-1] = Global_cls
    new_model = keras.models.Model(inputs=model.inputs, outputs=model.outputs)

    # for layer in new_model.layers:
        # layer.trainable = False

    # train_layers = ['D6', 'D7', 'D7_pool', 'global_3cls']

    # for trainable_layer in train_layers:
        # new_model.get_layer(trainable_layer).trainable = True


    new_model.compile(
        loss={
            'regression'    : keras_retinanet.losses.smooth_l1(),
            'classification': keras_retinanet.losses.focal(),
            'global_3cls': keras_retinanet.losses.classes_focal() 
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001),
        metrics=['accuracy']
    )

    # print model summary
    print(new_model.summary())


    train_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
    )
    val_image_data_generator = keras.preprocessing.image.ImageDataGenerator()
    
    # create a generator for training data
    train_generator = PascalVocGenerator(
        args.voc_path,
        'trainval',
        train_image_data_generator,
        batch_size=args.batch_size
    )
    # create a generator for testing data
    val_generator = PascalVocGenerator(
        args.voc_path,
        'test',
        val_image_data_generator,
        batch_size=args.batch_size
    )
    # start training
    # while(1):
        # train_generator.next()

    new_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator.image_names) // args.batch_size,
        epochs=50,
        verbose=1,
        validation_data=val_generator,
        validation_steps= len(val_generator.image_names) // args.batch_size,
        callbacks=[
            keras.callbacks.ModelCheckpoint('./data/snapshots/class3retinanet_resnet50_{epoch:02d}-{val_loss:.5f}.h5', monitor='val_loss', verbose=1, save_best_only=False),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0),
            csv_logger,
        ],
    )

    # store final result too
    new_model.save('.data/snapshots/class3retinanet_resnet50_voc_final.h5')
