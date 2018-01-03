#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: tf_record_db.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   create date: 2017/12/29
#   describtion:
#
#================================================================

import os
import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array


class TfRecordDb(object):
    """read and write database to tensorflow record"""

    def __init__(self,
                 label_file,
                 prefix,
                 record_save_path,
                 num_classes=3,
                 image_shape=(224, 224)):
        super(TfRecordDb, self).__init__()
        self.label_file = label_file
        self.prefix = prefix
        self.record_save_path = record_save_path
        self.num_classes = num_classes
        self.image_shape = (224, 224)
        self.broken_images = {}

    def parse_label_file(self, label_file):
        """parse label file to addrs and labels

        #Arguments
        label_file: String. file name contains label. each line is of format "file_abs_path label_id", e.g "/path/to/train_1.jpg 0"

        #Return
        return the addrs and labels
        addrs: List.
        labels: List.

        """
        addrs = []
        labels = []
        with open(label_file, 'r') as fh:
            images = fh.readlines()
            np.random.shuffle(images)
            for line in images:
                addr, label = line.split()
                addrs.append(addr)
                labels.append(int(label))
        return addrs, labels

    def write_record(self):
        """write db into record

        #Arguments
        label_file: String. file contains the image path and corresponding label id.
        record_file_name: String. path to save the tf record.

        """
        label_file = self.label_file
        record_file_path = self.record_save_path
        prefix = self.prefix
        addrs, labels = self.parse_label_file(label_file)
        record_file_directory = os.path.dirname(record_file_path)
        writer = tf.python_io.TFRecordWriter(self.record_save_path)
        if not os.path.exists(record_file_directory):
            os.makedirs(record_file_directory)
        for i in range(len(addrs)):
            # print how many images are saved every 1000 images
            if not i % 1000:
                print 'data: {}/{}'.format(i, len(addrs))
                sys.stdout.flush()
            # Load the image
            try:
                img = self.load_image(addrs[i], image_shape=self.image_shape)
                label = labels[i]
                # Create a feature
                feature = {
                    '{}/label'.format(prefix):
                    self._int64_feature(label),
                    '{}/image'.format(prefix):
                    self._bytes_feature(tf.compat.as_bytes(img.tostring()))
                }
                # Create an example protocol buffer
                example = tf.train.Example(
                    features=tf.train.Features(feature=feature))

                # Serialize to string and write on the file
                writer.write(example.SerializeToString())
            except Exception, e:
                self.broken_images.update({addrs[i]: 1})
        writer.close()
        sys.stdout.flush()
        print('num image broken: {}'.format(len(self.broken_images.keys())))

    def read_record(self, batch_size, endless=True):
        """docstring for read_record"""
        x_key = '{}/image'.format(self.prefix)
        y_key = '{}/label'.format(self.prefix)
        feature = {
            x_key: tf.FixedLenFeature([], tf.string),
            y_key: tf.FixedLenFeature([], tf.int64)
        }
        # Create a list of filenames and pass it to a queue
        if endless:
            num_epochs = None
        else:
            num_epochs = 1
        filename_queue = tf.train.string_input_producer(
            [self.record_save_path], num_epochs=num_epochs)
        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        # Decode the record read by the reader
        features = tf.parse_single_example(
            serialized_example, features=feature)
        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features[x_key], tf.uint8)
        # Cast label data into int32
        label = tf.cast(features[y_key], tf.int32)
        label = tf.one_hot(label, self.num_classes)
        # Reshape image data into the original shape
        image = tf.reshape(image, self.image_shape + (3, ))
        image = tf.cast(image, tf.float32)

        # Any preprocessing here ...

        # Creates batches by randomly shuffling tensors
        images, labels = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            capacity=10240,
            num_threads=6,
            min_after_dequeue=64)
        return images, labels

    def load_image(self, addr, image_shape):
        # read an image and resize to (224, 224)
        # cv2 load images as BGR, convert it to RGB
        """
        read an image and resize
        cv2 load images as BGR, convert it to RGB

        #Arguments
        addr: String.image path.
        image_shape: 2d tuple. reshape image to the size
        """
        # img = load_img(addr, target_size=image_shape)
        # x = np.asarray(img, dtype=np.float32)
        # x[..., 0] -= 103.939
        # x[..., 1] -= 116.779
        # x[..., 2] -= 123.68
        # return x
        img = cv2.imread(addr)
        img = cv2.resize(img, image_shape, interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.uint8)
        return img

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == '__main__':
    import sys
    file_name = sys.argv[1]
    save_file_path = sys.argv[2]
    prefix = sys.argv[3]
    tf_record_db = TfRecordDb(
        label_file=file_name, prefix=prefix, record_save_path=save_file_path)
    tf_record_db.write_record()
