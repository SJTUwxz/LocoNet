#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: evaluate_tf_record_callback.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   create date: 2017/12/29
#   describtion:
#
#================================================================

import keras
import numpy as np
import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from keras.callbacks import Callback


class EvaluateTfRecordCallback(Callback):
    """evaluate model on tf records"""

    def __init__(self,
                 model,
                 tf_record_db,
                 session,
                 checkout_save_path=None,
                 log_save_path=None,
                 post_process=None,
                 batch_size=32):
        super(EvaluateTfRecordCallback, self).__init__()
        self.original_model = model
        self.tf_record_db = tf_record_db
        self.session = session
        self.checkout_save_path = checkout_save_path
        self.log_save_path = log_save_path
        self.post_process = post_process
        self.batch_size = batch_size

        # build eval model
        eval_input = Input(shape=self.tf_record_db.image_shape + (3, ))
        eval_model = model(eval_input)
        eval_model = Model(inputs=eval_input, outputs=eval_model)
        eval_model.compile(
            optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        self.eval_model = eval_model

        # eval x and y
        self.x = None
        self.y = None

    def on_train_begin(self, logs={}):
        #create log_save file
        self.csv_file = open(self.log_save_path, 'w')
        self.csv_file.write('epoch,loss,accuracy,val_loss,val_accurary\n')
        x, y = self.set_xy()
        self.x = x
        self.y = y

    def on_epoch_end(self, epoch, logs={}):

        # evaluate
        result = self.eval_model.evaluate(x=self.x, y=self.y, batch_size=32)
        # update logs
        logs.update({'epoch': epoch})
        logs.update({'val_loss': result[0]})
        logs.update({'val_acc': result[1]})
        self.csv_file.write(
            '{epoch:02d},{loss:.5f},{acc:.5f},{val_loss:.5f},{val_acc:.5f}\n'.
            format(**logs))
        print(
            '\nepoch: {epoch:02d},loss: {loss:.5f},acc: {acc:.5f},val_loss: {val_loss:.5f},val_acc: {val_acc:.5f}\n'.
            format(**logs))
        #save checkpoint
        self.original_model.save_weights(
            self.checkout_save_path.format(**logs))

    def set_xy(self):
        """set x and y"""
        tfrecords_filename = self.tf_record_db.record_save_path
        prefix = self.tf_record_db.prefix
        num_classes = self.tf_record_db.num_classes
        record_iterator = tf.python_io.tf_record_iterator(
            path=tfrecords_filename)
        #get data
        x_key = '{}/image'.format(self.tf_record_db.prefix)
        y_key = '{}/label'.format(self.tf_record_db.prefix)
        imgs = []
        labels = []
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            img_string = (example.features.feature[x_key].bytes_list.value[0])
            label = int(example.features.feature[y_key].int64_list.value[0])
            img_1d = np.fromstring(img_string, dtype=np.uint8)
            img = np.reshape(img_1d, self.tf_record_db.image_shape + (3, ))
            imgs.append(img)
            labels.append(label)

        imgs = np.array(imgs)
        imgs = np.asarray(imgs, dtype=np.float32)
        labels = keras.utils.to_categorical(labels, num_classes=num_classes)
        if self.post_process:
            for f in self.post_process:
                imgs = f(imgs)
        return imgs, labels
