#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: tfrecords_db.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/01/02
#   description:
#
#================================================================

import os
import cv2
import sys
import types
import numpy as np
import tensorflow as tf


class TfRecordDB(object):
    """database of tensorflow record. Read and write tensorlfow record"""

    def __init__(self,
                 label_file,
                 prefix,
                 record_save_path,
                 post_processors=None,
                 num_classes=3,
                 image_shape=(224, 224),
                 channel=3):
        """construct function

        Args:
            label_file: file name contains label. each line is of format "file_abs_path label_id", e.g "/path/to/train_1.jpg 0"
            prefix: the prefix of record file. eg. 'train', 'val'
            record_save_path: the save path of record file

        Kwargs:
            post_processors (Optional): List or one function. functions to process the input tensors(x,y). The inputs of theses functions are (x, y). x and y are tensorflow tensors or numpy arrays. Make sure your functions support both types.
            num_classes (Optional): the number of classes. Default to 3
            image_shape (Optional): the image shape. Default to (224,224)
            channel (Optional): the image channel. Default is 3


        """
        self._label_file = label_file
        self._prefix = prefix
        self._record_save_path = record_save_path
        self._post_processors = post_processors
        self._num_classes = num_classes
        self._image_shape = image_shape
        self._channel = channel

        self.x_key = '{}/image'.format(self._prefix)
        self.y_key = '{}/label'.format(self._prefix)

        self.broken_images = {}

        if self._post_processors != None:
            assert isinstance(
                self._post_processors, (list, tuple, types.FunctionType)
            ), 'post_processors must be a function or a list of functions'

    def get_input_shape(self):
        """return the input shape
        Returns: input shape

        """
        return self._image_shape + (self._channel, )

    def get_output_shape(self):
        """return the output shape
        Returns: output shape

        """
        return self._num_classes

    def get_record_iterator(self):
        """return the record iterator
        Returns: @tf.python_io.tf_record_iterator

        """
        if not os.path.isfile(self._record_save_path):
            self.write_record()
        return tf.python_io.tf_record_iterator(path=self._record_save_path)

    def get_steps(self, batch_size):
        """get steps when minibatch-size is batch_size

        Args:
            batch_size: data length each step

        Returns: steps

        """
        addrs, labels = self._parse_label_file(self._label_file)
        return len(addrs) // batch_size

    def _parse_label_file(self, label_file):
        """parse label file to addrs and labels

        Args:
            label_file: file name contains label. each line is of format "file_abs_path label_id", e.g "/path/to/train_1.jpg 0"

        Returns: the addrs and labels. each is a list and they has the same length
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

    def _apply_post_processors(self, img, label):
        """post processes during @self.read_example or @self.read_record
        Returns: post processed img and label

        """
        if self._post_processors == None:
            return img, label
        elif isinstance(self._post_processors, (list, tuple)):
            for func in self._post_processors:
                img, label = func(img, label)
        else:
            img, label = self._post_processors(img, label)

        return img, label

    def read_example(self, string_record):
        """unserialize the string_record to object @tf.train.Example

        Args:
            string_record: serialized example

        Returns:
            img: the unserialized image numpy array
            label: the label index

        """
        example = tf.train.Example()
        example.ParseFromString(string_record)
        img_string = (example.features.feature[self.x_key].bytes_list.value[0])
        label = int(example.features.feature[self.y_key].int64_list.value[0])
        img_1d = np.fromstring(img_string, dtype=np.uint8)
        img = np.reshape(img_1d, self.get_input_shape())
        img = np.asarray(img, dtype=np.float32)
        return self._apply_post_processors(img, label)

    def write_example(self, img, label):
        """produce an instance of @tf.train.Example from @img and @label

        Args:
            img: the unserialized image numpy array
            label: the label index

        Returns:
            an instance of @tf.train.Example

        """
        feature = {
            self.x_key: self._bytes_feature(
                tf.compat.as_bytes(img.tostring())),
            self.y_key: self._int64_feature(label)
        }
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example

    def write_record(self):
        """write the database to tf record
        Returns: None

        """
        label_file = self._label_file
        record_file_path = self._record_save_path
        prefix = self._prefix
        addrs, labels = self._parse_label_file(label_file)
        record_file_directory = os.path.dirname(record_file_path)
        if not os.path.exists(record_file_directory):
            os.makedirs(record_file_directory)
        writer = tf.python_io.TFRecordWriter(self._record_save_path)
        for i in range(len(addrs)):
            # print how many images are saved every 1000 images
            if not i % 1000:
                print 'data: {}/{}'.format(i, len(addrs))
                sys.stdout.flush()
            # Load the image
            try:
                img = self._load_image(addrs[i], image_shape=self._image_shape)
                label = labels[i]
                example = self.write_example(img, label)
                # Serialize to string and write on the file
                writer.write(example.SerializeToString())
            except Exception, e:
                self.broken_images.update({addrs[i]: 1})
        writer.close()
        sys.stdout.flush()
        print('num image broken: {}'.format(len(self.broken_images.keys())))

    def read_record(self,
                    batch_size,
                    capacity=10240,
                    num_threads=4,
                    min_after_dequeue=64):
        """read the databse and return the input tensor
        Args:
            batch_size: the input batch_size

        Kwargs:
            capacity (Optional) : An integer. The maximum number of elements in the queue. Default to 10240
            num_threads (Optional): The number of threads enqueuing tensor_list. Default to 4
            min_after_dequeue (Optional): Minimum number elements in the queue after a dequeue, used to ensure a level of mixing of elements. Default to 64

        Returns:
            the input tensors (x,y) . x represents the img data and y is the labels.

        """
        if not os.path.isfile(self._record_save_path):
            print('tf record has not been generated. generating now...')
            print('record file path: {}'.format(self._record_save_path))
            self.write_record()
        x_key = '{}/image'.format(self._prefix)
        y_key = '{}/label'.format(self._prefix)
        feature = {
            x_key: tf.FixedLenFeature([], tf.string),
            y_key: tf.FixedLenFeature([], tf.int64)
        }
        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer(
            [self._record_save_path], num_epochs=None)
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
        # Reshape image data into the original shape
        image = tf.reshape(image, self.get_input_shape())
        image = tf.cast(image, tf.float32)

        # Any preprocessing here ...

        # Creates batches by randomly shuffling tensors
        images, labels = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            capacity=capacity,
            num_threads=num_threads,
            min_after_dequeue=min_after_dequeue)
        return self._apply_post_processors(images, labels)

    @staticmethod
    def _load_image(addr, image_shape):
        """read an image and resize to image_shape. cv2 load images as BGR, convert it to RGB

        Args:
            addr: image path
            image_shape: 2d tuple (height, width) . reshape image to the size

        Returns: img numpy array

        """
        img = cv2.imread(addr)
        img = cv2.resize(img, image_shape, interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.uint8)
        return img

    @staticmethod
    def _int64_feature(value):
        """int64 feature. convert int64 Interger to @tf.train.Feature

        Args:
            value: an Integer. label index

        Returns: an instance of @tf.train.Feature

        """
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        """bytes feature. convert bytes to @tf.train.Feature

        Args:
            value: bytes data.

        Returns: an instance of @tf.train.Feature

        """
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
