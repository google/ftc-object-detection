# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import cv2
import numpy as np
import os
from tensorflow.contrib.lite.python.interpreter import Interpreter

TOP_CLASS_SCORE_THRESHOLD = .6
Y_SCALE = 10.0
X_SCALE = 10.0
W_SCALE = 10.0
H_SCALE = 10.0


def ExpIt(val):
  return 1. / (1. + np.exp(-val))


def ParseBoxPriors(priors_filename):
  box_priors = []
  for x in open(priors_filename):
    x = x.strip()
    if x:
      nums = x.split(" ")
      box_priors.append(map(float, nums))
  box_priors = np.array(box_priors)
  assert len(box_priors) == 4
  return box_priors

def ParseLabels(labels_filename):
  labels = {}
  for i, line in enumerate(open(labels_filename)):
    labels[i] = line.strip()
  return labels


class ObjectDetector:
  def __init__(self, model_filename, labels_filename, box_priors_filename):
    self.interp = Interpreter(model_filename)
    self.inputs = self.interp.get_input_details()
    self.outputs = self.interp.get_output_details()
    self.interp.allocate_tensors()
    self.labels = ParseLabels(labels_filename)
    self.box_priors = ParseBoxPriors(box_priors_filename)

  def detect(self, image):
    resized_image = cv2.resize(image[0,:,:,:], (300, 300))
    expanded_image = np.expand_dims(resized_image, axis=0)
    normalized_image = expanded_image.astype(np.float32) / 128. - 1.

    self.interp.set_tensor(self.inputs[0]["index"], normalized_image)
    self.interp.invoke()
    output_locations = self.interp.get_tensor(self.outputs[0]["index"])
    output_classes = self.interp.get_tensor(self.outputs[1]["index"])
    num_results = output_classes.shape[1]
    print("Num results:", num_results)
    boxes = []
    for i in range(num_results):
      ycenter = (
          output_locations[0, i, 0] / Y_SCALE * self.box_priors[2, i] +
          self.box_priors[0, i])
      xcenter = (
          output_locations[0, i, 1] / X_SCALE * self.box_priors[3, i] +
          self.box_priors[1, i])
      h = math.exp(output_locations[0, i, 2] / H_SCALE) * self.box_priors[2, i]
      w = math.exp(output_locations[0, i, 3] / W_SCALE) * self.box_priors[3, i]
      ymin = ycenter - h / 2
      xmin = xcenter - w / 2
      ymax = ycenter + h / 2
      xmax = xcenter + w / 2

      output_locations[0, i, 0] = ymin
      output_locations[0, i, 1] = xmin
      output_locations[0, i, 2] = ymax
      output_locations[0, i, 3] = xmax

      exp_scores = ExpIt(output_classes[0, i])
      top_class_score_index = np.argmax(exp_scores[1:]) + 1
      if exp_scores[top_class_score_index] > TOP_CLASS_SCORE_THRESHOLD:
        rectf = output_locations[0, i, [1, 0, 3, 2]]
        # Not actually a probability?
        value = exp_scores[top_class_score_index]
        boxes.append((value, rectf, top_class_score_index))

    boxes.sort(key=lambda x: x[0])

    output_dict = {
        'num_detections': len(boxes),
        'detection_classes': [box[2] for box in boxes],
        'detection_boxes': [box[1] for box in boxes],
        'detection_scores': [box[0] for box in boxes],
        }
    return output_dict
