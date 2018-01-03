#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: tfrecord_eval.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/01/02
#   description:
#
#================================================================

import os
import sys
import keras
import numpy as np
import tensorflow as tf
from multiprocessing import Process, Queue,Value
from keras.layers import Input
from keras.models import Model
from keras.callbacks import Callback

CURRENT_FILE_DIRECTORY = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(CURRENT_FILE_DIRECTORY, '../'))

from tfrecords_db import TfRecordDB


class TfRecordEvalCallback(Callback):
    """evaluate callback for tf records database"""

    def __init__(self,
                 model,
                 tf_record_db,
                 checkpoint_save_path=None,
                 log_save_path=None,
                 batch_size=32):
        """
        Args:
            model: instance of @keras.model.Model. The model to be evaluate. The model's input must be a @keras.layers.Input layer.
            tf_record_db: instance of  @tfrecords_db.TfRecordDB.

        Kwargs:
            checkpoint_save_path: file to save the checkpoint.
            log_save_path: file to save the log file.
            batch_size: the evaluation batch size.


        """
        Callback.__init__(self)

        self._model = model
        self._tf_record_db = tf_record_db
        self._checkpoint_save_path = checkpoint_save_path
        self._log_save_path = log_save_path
        self._batch_size = batch_size

        self._data_queue = Queue()
        self._is_done = Value('d', 0)

    def _data_producer(self):
        """produce the evaluate examples and put into ＠self._data_queue
        Returns: None

        """
        imgs = []
        labels = []
        for string_record in self._tf_record_db.get_record_iterator():
            img, label = self.tf_record_db.read_example(string_record)
            imgs.append(img)
            labels.append(label)
            if len(imgs) == self._batch_size:
                img_array = np.array(imgs)
                img_array = np.asarray(img_array, dtype=np.float32)
                label_array = np.array(labels)
                self._data_queue.put([img_array, label_array])
                imgs = []
                labels = []
            if len(labels) != 0:
                img_array = np.array(imgs)
                img_array = np.asarray(img_array, dtype=np.float32)
                label_array = np.array(labels)
                self._data_queue.put([img_array, label_array])
        self._is_done.value = 1

    def _evaluate(self):
        """evaluate @self._tf_record_db
        Returns: return the metrics
        """
        results = None
        num_samples = 0
        producer = Process(target=self._data_producer)
        producer.start()
        while self._is_done.value == 0:
            imgs, labels = self._data_queue.get()
            batch_result = self._model.evaluate(x=imgs, y=labels)
            num_samples += len(labels)
            if results == None:
                results = len(labels) * batch_result
            else:
                results += len(labels) * batch_result
        results /= num_samples
        return results

    def on_train_begin(self, logs={}):
        """create log save file on train begin

        Kwargs:
            logs: contains the metrics of the model such as loss,acc


        """
        self.csv_file = open(self._log_save_path, 'w')
        self.csv_file.write('epoch,loss,accuracy,val_loss,val_accurary\n')

    def on_epoch_end(self, epoch, logs={}):
        """evaluate the model and save the checkpoint and log on epoch end

        Args:
            epoch: training epoch

        Kwargs:
            logs: contains the metrics of the model such as loss,acc

        """
        result = self._evaluate()
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
        self._model.save_weights(self._checkpoint_save_path.format(**logs))
