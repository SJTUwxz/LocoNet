#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: export_resnet50.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/01/08
#   description:
#
#================================================================

from training.common import preprocess_input, preprocess_output

from keras_extra.applications.resnet50 import ResNet50


def save_to_json(weights, save_path):
    """TODO: Docstring for save_to_json.
    Returns: TODO

    """
    num_classes = 3
    input_shape = (224, 224, 3)
    batch_size = 32

    preprocess_input_func = preprocess_input
    preprocess_output_func = preprocess_output(num_classes)

    model = ResNet50(
        num_classes=num_classes,
        input_shape=input_shape,
        batch_size=batch_size,
        preprocess_input=preprocess_input_func,
        preprocess_output=preprocess_output_func,
        weights=weights)

    _model = model._model
    model_json = _model.to_json()
    with open(save_path, "w") as json_file:
        json_file.write(model_json)


if __name__ == "__main__":
    weights = './data/snapshots/007-resnet50-on-full-db/20180107-023834/snapshot_00-0.13984--0.94756.h5'
    save_path = './data/milestone/resnet50_cls3/resnet50_backbone.json'
    save_to_json(weights, save_path)
