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
import keras_retinanet


def classes_focal(alpha=0.44, gamma=2.0):
    def _classes_focal(y_true, y_pred):
        alpha = 0.44
        gamma = 2.0
        # discard batches, throw all labels / classifications on one big blob
        # labels         = keras.backend.reshape(y_true, (-1, keras.backend.shape(y_true)))
        # classes = keras.backend.reshape(y_pred, (-1, keras.backend.shape(y_pred)))
        labels = y_true
        classes = y_pred

        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = keras_retinanet.backend.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = keras_retinanet.backend.where(keras.backend.equal(labels, 1), 1 - classes, classes)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classes)
        cls_loss = keras.backend.sum(cls_loss)

        # cls_loss = cls_loss / (keras.backend.maximum(1.0, keras.backend.sum(anchor_state)))
        return cls_loss
    return _classes_focal

def focal(alpha=0.25, gamma=2.0):
    def _focal(y_true, y_pred):
        # discard batches, throw all labels / classifications on one big blob
        labels         = keras.backend.reshape(y_true, (-1, keras.backend.shape(y_true)[2]))
        classification = keras.backend.reshape(y_pred, (-1, keras.backend.shape(y_pred)[2]))

        # filter out "ignore" anchors
        anchor_state   = keras.backend.max(labels, axis=1)  # -1 for ignore, 0 for background, 1 for object
        indices        = keras_retinanet.backend.where(keras.backend.not_equal(anchor_state, -1))
        classification = keras_retinanet.backend.gather_nd(classification, indices)
        labels         = keras_retinanet.backend.gather_nd(labels, indices)
        anchor_state   = keras_retinanet.backend.gather_nd(anchor_state, indices)

        # select classification scores for labeled anchors
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = keras_retinanet.backend.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = keras_retinanet.backend.where(keras.backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)
        cls_loss = keras.backend.sum(cls_loss)

        # "The total focal loss of an image is computed as the sum
        # of the focal loss over all ~100k anchors, normalized by the
        # number of anchors assigned to a ground-truth box."
        cls_loss = cls_loss / (keras.backend.maximum(1.0, keras.backend.sum(anchor_state)))
        return cls_loss

    return _focal


def smooth_l1(sigma=3.0):
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        # discard batches, throw all regression / anchor states on one big blob
        regression        = keras.backend.reshape(y_pred, (-1, 4))
        regression_target = keras.backend.reshape(y_true[:, :, :4], (-1, 4))
        anchor_state      = keras.backend.reshape(y_true[:, :, 4], (-1,))

        # filter out "ignore" anchors
        indices           = keras_retinanet.backend.where(keras.backend.equal(anchor_state, 1))
        regression        = keras_retinanet.backend.gather_nd(regression, indices)
        regression_target = keras_retinanet.backend.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = keras_retinanet.backend.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )
        regression_loss = keras.backend.sum(regression_loss)

        divisor         = keras.backend.maximum(keras.backend.shape(indices)[0], 1)
        divisor         = keras.backend.cast(divisor, keras.backend.floatx())
        return regression_loss / divisor

    return _smooth_l1
