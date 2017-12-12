
# coding: utf-8

# In[1]:


import keras
import keras.preprocessing.image
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.preprocessing.pascal_voc import PascalVocGenerator

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

import tensorflow as tf
import xml.etree.ElementTree as ET

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
keras.backend.tensorflow_backend.set_session(get_session())


# ## Load RetinaNet model

# In[ ]:


model = keras.models.load_model('snapshots/resnet50_voc_best.h5', custom_objects=custom_objects)
#print(model.summary())


# ## Initialize data generators

# In[3]:


# create image data generator object
val_image_data_generator = keras.preprocessing.image.ImageDataGenerator()

# create a generator for testing data
val_generator = PascalVocGenerator(
   '/data/users/xiziwang/tools/nsp/JHdevkit/VOC2007',
   'test',
   val_image_data_generator,
   batch_size=1,
)
index = 0


# ## Run detection on example

# In[6]:


# load image
f = open('/home/xiziwang/tools/test_erotic_images.txt','r')
files = f.read().splitlines()
result = open('/home/xiziwang/tools/test_result_global/result.txt','w')

for fn in files:

  # image = val_generator.load_image(fn) #load_image(index)
  image = cv2.imread(fn)

# copy to draw on
  draw = image.copy()
  draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
  filename = fn.split('/')[-1].strip('.jpg')
# preprocess image for network
  image = val_generator.preprocess_image(image)
  image, scale = val_generator.resize_image(image)
  #xmlfilename = os.path.join('/data/users/xiziwang/tools/nsp/JHdevkit/VOC2007/Annotations',filename+'.xml')
  #annotations = val_generator.load_annotations2(xmlfilename)
  index += 1

# process image
  start = time.time()
  _, _, detections, classes = model.predict_on_batch(np.expand_dims(image, axis=0))
  print("processing time: {}s".format(time.time() - start) )
# compute predicted labels and scores
  predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
  scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]
  if classes[0][0] > classes[0][1]:
    jhclass = 'sexy'
  else:
    jhclass = 'erotic'
# correct for image scale
  detections[:, :4] /= scale

# visualize detections
  for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
    if score < 0.4:
      continue
    b = detections[0, idx, :4].astype(int)
    cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 3)
    caption = "{} {:.3f}".format(val_generator.label_to_name(label), score)
    cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
  result.write('{} {}\n'.format(fn, jhclass))
# visualize annotations
  #for annotation in annotations:
  #  label = int(annotation[4])
  #  b = annotation[:4].astype(int)
  #  cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
  #  caption = "{}".format(val_generator.label_to_name(label))
  #  cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 1)
  #  cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 1)
    
  # plt.figure(figsize=(15, 15))
  # plt.axis('off')
  # plt.imshow(draw)
  #plt.show()
  filename = fn.split('/')[-1]
  # plt.savefig('/home/xiziwang/tools/test_result/sexy/'+filename)
  cv2.imwrite('/home/xiziwang/tools/test_result_global/erotic/'+filename, draw) 

