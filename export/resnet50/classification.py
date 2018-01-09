#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: classification.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/01/08
#   description:
#
#================================================================

import os
import cv2
import sys
import keras
import shutil
import numpy as np

IMAGE_SHAPE = (224, 224)


def load_image(img_addr):
    """load and preprocess image

    Args:
        img_addr (TODO): TODO

    Returns: TODO

    """

    # image with BGR format
    img = cv2.imread(img_addr)
    img = cv2.resize(img, IMAGE_SHAPE, interpolation=cv2.INTER_CUBIC)
    img = np.asarray(img, dtype=np.float32)
    #substract mean
    img -= [103.939, 116.779, 123.68]
    return img


def load_model(model_json_file, weights):
    """TODO: Docstring for load_model.

    Args:
        model_json_file (TODO): TODO
        weights (TODO): TODO

    Returns: TODO

    """
    json_file = open(model_json_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights(weights)
    print('model successful loaded.')
    return model


def classification(model, image_dir, output_dir):
    """TODO: Docstring for classification.

    Args:
        model (TODO): TODO
        image_dir (TODO): TODO

    Returns: TODO

    """
    result_dir = [
        os.path.join(output_dir, dirname)
        for dirname in ['normal', 'sexy', 'porn', 'broken']
    ]
    for dir_name in result_dir:
        try:
            os.makedirs(dir_name)
        except:
            pass

    for (dirpath, dirnames, filenames) in os.walk(image_dir):
        for filename in filenames:
            image_path = os.path.join(dirpath, filename)
            try:
                img = load_image(image_path)
                img = np.expand_dims(img, axis=0)
                result = model.predict(img)[0]
                cls_index = np.argmax(result)
                print('{} :{}'.format(image_path, cls_index))
            except Exception, e:
                print('image {} broken: {}'.format(image_path, e))
                cls_index = -1
            dest_path = os.path.join(result_dir[cls_index], filename)
            shutil.copyfile(image_path, dest_path)


if __name__ == "__main__":
    model_json_file = './data/milestone/resnet50_cls3/resnet50_backbone.json'
    weights = './data/milestone/resnet50_cls3/snapshot_00-0.13984--0.94756.h5'
    model = load_model(model_json_file, weights)

    src_dir = sys.argv[1]
    output_dir = sys.argv[2]
    classification(model, src_dir, output_dir)
