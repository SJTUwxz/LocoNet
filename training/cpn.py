#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: cpn.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/01/09
#   description:
#
#================================================================

import keras
from keras_extra.applications.base_model import BaseModel
from keras_retinanet.models.jh_resnet import ResNet50RetinaNet


class CPN(BaseModel):
    """cpn"""

    def __init__(self, *args, **kwargs):
        """TODO: to be defined1.

        Args:
            *args (TODO): TODO
            **kwargs (TODO): TODO


        """
        BaseModel.__init__(self, *args, **kwargs)

    def _setup_model(self):
        """TODO: Docstring for _setup_model.
        Returns: TODO

        """
        fix_layers = False
        # weights = './data/snapshots/001-retinanet/resnet50_05-0.39389.h5'
        weights = 'data/snapshots/010-stage1-add-normal/class3retinanet_resnet50_02-0.56798.h5'

        image = keras.layers.Input(self._input_shape)
        model = ResNet50RetinaNet(image, num_classes=10, weights=weights)
        D7_pool = model.get_layer('D7_pool').output
        Global_cls = keras.layers.Dense(
            self._num_classes, activation='softmax',
            name='global_3cls')(D7_pool)
        new_model = keras.models.Model(
            inputs=model.inputs, outputs=[Global_cls])
        new_model.load_weights(weights, by_name=True)

        if fix_layers:
            for layer in new_model.layers:
                layer.trainable = False
            train_layers = ['D6', 'D7', 'D7_pool', 'global_3cls']
            for trainable_layer in train_layers:
                new_model.get_layer(trainable_layer).trainable = True
        return new_model
