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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, log_loss
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

        self._model.compile(
            loss='categorical_crossentropy',
            optimizer='sgd',
            metrics=['accuracy'])

    def _evaluate(self):
        """evaluate @self._tf_record_db
        Returns: return the metrics
        """

        def _pred_batch(imgs, labels):
            img_array = np.stack(imgs)
            batch_pred_y = self._model.predict(img_array)
            return batch_pred_y

        batch_imgs = []
        batch_labels = []
        labels = []
        pred_y = []
        results = None
        num_samples = 0
        for string_record in self._tf_record_db.get_record_iterator():
            img, label = self._tf_record_db.read_example(string_record)
            batch_imgs.append(img)
            labels.append(label)
            batch_labels.append(label)
            if len(batch_imgs) == self._batch_size:
                batch_pred_y = _pred_batch(batch_imgs, batch_labels)
                pred_y.extend(batch_pred_y)
                batch_imgs = []
                batch_labels = []
        if len(batch_labels) != 0:
            batch_pred_y = _pred_batch(batch_imgs, batch_labels)
            pred_y.extend(batch_pred_y)
        numerical_y = np.argmax(labels, axis=1)
        numerical_pred_y = np.argmax(pred_y, axis=1)
        loss = log_loss(numerical_y, pred_y)
        acc = accuracy_score(numerical_y, numerical_pred_y)
        class_report = classification_report(numerical_y, numerical_pred_y)
        cm = confusion_matrix(numerical_y, numerical_pred_y)
        return {
            'val_loss': loss,
            'val_acc': acc,
            'class_report': class_report,
            'cm': cm
        }

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
        logs.update(result)
        self.csv_file.write(
            '{epoch:02d},{loss:.5f},{acc:.5f},{val_loss:.5f},{val_acc:.5f}\n'.
            format(**logs))
        print(
            '\nepoch: {epoch:02d},loss: {loss:.5f},acc: {acc:.5f},val_loss: {val_loss:.5f},val_acc: {val_acc:.5f}\n{class_report}\n{cm}\n'.
            format(**logs))
        #save checkpoint
        self._model.save_weights(self._checkpoint_save_path.format(**logs))
