#!/usr/bin/env python

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras
import keras.preprocessing.image
from keras.preprocessing import image
from sklearn.metrics import classification_report
from keras_retinanet.utils.image import preprocess_image, resize_image, random_transform
from keras_retinanet.preprocessing.pascal_voc import PascalVocGenerator
from keras_retinanet.utils.class3_eval import evaluate_jh
from keras_retinanet.models.jh_resnet import custom_objects
from keras_retinanet.utils.keras_version import check_keras_version
import cv2
import tensorflow as tf

import argparse
from preprocess.MyImageGenerator import MyImageDataGenerator

import numpy as np
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def parse_args():
    parser = argparse.ArgumentParser(description='Simple training script for COCO object detection.')
    parser.add_argument('model', help='Path to RetinaNet model.')
    parser.add_argument('jh_path', help='Path to sensitive image directory (ie. /tmp/COCO).')
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--set', help='Name of the set file to evaluate (defaults to val2017).', default='val2017')
    parser.add_argument('--score-threshold', help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)

    return parser.parse_args()

if __name__ == '__main__':
    # parse arguments
    args = parse_args()
#     image_data_generator = MyImageDataGenerator(
#         featurewise_center=False,
#         samplewise_center=False,
#         featurewise_std_normalization=False,
#         samplewise_std_normalization=False,
#         zca_whitening=False,
#         )

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # create the model
    print('Loading model, this may take a second...')
    model = keras.models.load_model(args.model, custom_objects=custom_objects)

    # create image data generator object
    test_label = './data/labels/labeled_10w_test.txt'
    # test_gen = image_data_generator.flow_from_label_file(
            # test_label,
            # phase = 'val',
            # batch_size=32,
            # is_ergodic_files=None,
            # balance = False
            # ) 
    # test_steps = test_gen.steps_per_epoch()

    # eval = model.evaluate_generator(test_gen, 32) 
    # print eval
    y_true = []
    y_pred = []
    f = open(test_label,'r')
    for k,line in enumerate(f.read().splitlines()):
        fname, label = line.split(' ') 
        # img = cv2.imread(fname)
        # x = preprocess_image(img) 
        # x, image_scale = resize_image(x)
        img = image.load_img(fname, target_size=(299,299))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis=0)
        x = x / 255.
        print np.size(x) 
        preds = model.predict(x)
        pred = np.argmax(preds[0])
        y_pred.append(pred) 
        y_true.append(int(label))
        print pred, label

    print(classification_report(y_true, y_pred, target_names=['normal','sexy','erotic'])) 
