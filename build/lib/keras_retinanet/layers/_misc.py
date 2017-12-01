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
import keras_retinanet.backend
import keras_retinanet.utils.anchors

import numpy as np


class Anchors(keras.layers.Layer):
    def __init__(self, size, stride, ratios=None, scales=None, *args, **kwargs):
        self.size   = size
        self.stride = stride
        self.ratios = ratios
        self.scales = scales

        if ratios is None:
            self.ratios  = np.array([0.5, 1, 2], keras.backend.floatx()),
        if scales is None:
            self.scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),

        self.num_anchors = len(ratios) * len(scales)
        self.anchors     = keras.backend.variable(keras_retinanet.utils.anchors.generate_anchors(
            base_size=size,
            ratios=ratios,
            scales=scales,
        ))

        super(Anchors, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        features = inputs
        features_shape = keras.backend.shape(features)[1:3]

        # generate proposals from bbox deltas and shifted anchors
        anchors = keras_retinanet.backend.shift(features_shape, self.stride, self.anchors)
        anchors = keras.backend.expand_dims(anchors, axis=0)

        return anchors

    def compute_output_shape(self, input_shape):
        if None not in input_shape[1:]:
            total = np.prod(input_shape[1:3]) * self.num_anchors
            return (input_shape[0], total, 4)
        else:
            return (input_shape[0], None, 4)

    def get_config(self):
        return {
            'size'   : self.size,
            'stride' : self.stride,
            'ratios' : self.ratios.tolist(),
            'scales' : self.scales.tolist(),
        }


class NonMaximumSuppression(keras.layers.Layer):
    def __init__(self, nms_threshold=0.4, top_k=None, max_boxes=300, *args, **kwargs):
        self.nms_threshold = nms_threshold
        self.top_k         = top_k
        self.max_boxes     = max_boxes
        super(NonMaximumSuppression, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        boxes, classification, detections = inputs

        # TODO: support batch size > 1.
        boxes          = boxes[0]
        classification = classification[0]
        detections     = detections[0]

        scores          = keras.backend.max(classification, axis=1)

        # selecting best anchors theoretically improves speed at the cost of minor performance
        if self.top_k:
            scores, indices = keras_retinanet.backend.top_k(scores, self.top_k, sorted=False)
            boxes           = keras.backend.gather(boxes, indices)
            classification  = keras.backend.gather(classification, indices)
            detections      = keras.backend.gather(detections, indices)

        indices = keras_retinanet.backend.non_max_suppression(boxes, scores, max_output_size=self.max_boxes, iou_threshold=self.nms_threshold)

        detections = keras.backend.gather(detections, indices)
        return keras.backend.expand_dims(detections, axis=0)

    def compute_output_shape(self, input_shape):
        return (input_shape[2][0], None, input_shape[2][2])

    def get_config(self):
        return {
            'nms_threshold' : self.nms_threshold,
            'top_k'         : self.top_k,
            'max_boxes'     : self.max_boxes,
        }


class UpsampleLike(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        return keras_retinanet.backend.resize_images(source, (target_shape[1], target_shape[2]))

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)


class RegressBoxes(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        anchors, regression = inputs
        return keras_retinanet.backend.bbox_transform_inv(anchors, regression)

    def compute_output_shape(self, input_shape):
        return input_shape[0]
