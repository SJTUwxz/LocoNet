#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: 107-resnet50-on-full-db.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/01/08
#   description:
#
#================================================================

# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
# from keras.backend.tensorflow_backend import set_session
# set_session(session)

import numpy as np

from training.model_trainer import run
from training.common import preprocess_input, preprocess_output

if __name__ == "__main__":
    import sys
    import time
    import logging
    logging.getLogger().setLevel(logging.INFO)
    run_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    branch = '009-stage1-on-10w'
    run_name = '{}/{}'.format(branch, run_time)
    logging.info('run_name: ' + run_name)

    train_label_file = './data/labels/10w_train.txt'
    val_label_file = './data/labels/10w_val.txt'
    test_label_file = './data/labels/10w_test.txt'

    train_record_path = './data/tf_records/10w_nsp/train.record'
    val_record_path = './data/tf_records/10w_nsp/val.record'
    test_record_path = './data/tf_records/10w_nsp/test.record'
    num_classes = 3
    input_shape = (224, 224, 3)
    batch_size = 32
    preprocess_input_func = preprocess_input
    preprocess_output_func = preprocess_output(num_classes)
    snapshot_save_path = './data/snapshots/{}'.format(
        run_name) + 'snapshot_{epoch:02d}-{val_loss:.5f}-{val_acc:.5f}.h5'
    log_save_path = './data/logs/{}'.format(run_name) + 'training_log.csv'
    weights = None
    epochs = 50

    run_type = sys.argv[1]
    predict_img_path = None
    eval_weights = None
    if len(sys.argv) == 3:
        if run_type == 'predict':
            predict_img_path = sys.argv[2]
        else:
            eval_weights = sys.argv[2]

    model_name = 'cpn'
    run(model_name, run_type, num_classes, input_shape, batch_size,
        preprocess_input_func, preprocess_output_func, weights,
        train_label_file, train_record_path, val_label_file, val_record_path,
        test_record_path, snapshot_save_path, log_save_path, epochs,
        eval_weights, predict_img_path)
