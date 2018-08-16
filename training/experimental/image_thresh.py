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

import numpy as np
import cv2
import sys

import pdb

# Create the window
window_name = 'All Shapes Image (thresholded)'
cv2.namedWindow(window_name)

im = cv2.imread(sys.argv[1])

converted = im
#  converted = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

# Get the channels
ch0 = converted[:, :, 0]
ch1 = converted[:, :, 1]
ch2 = converted[:, :, 2]

# Create the trackbars
cv2.createTrackbar('min_1', window_name, ch0.min(), ch0.max(), lambda x: None)
cv2.createTrackbar('max_1', window_name, ch0.min(), ch0.max(), lambda x: None)
cv2.createTrackbar('min_2', window_name, ch1.min(), ch1.max(), lambda x: None)
cv2.createTrackbar('max_2', window_name, ch1.min(), ch1.max(), lambda x: None)
cv2.createTrackbar('min_3', window_name, ch2.min(), ch2.max(), lambda x: None)
cv2.createTrackbar('max_3', window_name, ch2.min(), ch2.max(), lambda x: None)

while True:
    # Get the trackbar positions
    min_1 = cv2.getTrackbarPos('min_1', window_name)
    max_1 = cv2.getTrackbarPos('max_1', window_name)
    min_2 = cv2.getTrackbarPos('min_2', window_name)
    max_2 = cv2.getTrackbarPos('max_2', window_name)
    min_3 = cv2.getTrackbarPos('min_3', window_name)
    max_3 = cv2.getTrackbarPos('max_3', window_name)


    lower = np.array([min_1, min_2, min_3])
    upper = np.array([max_1, max_2, max_3])
    mask = cv2.inRange(converted, lower, upper)
    masked = cv2.bitwise_and(im, im, mask=mask)

    cv2.imshow(window_name, mask)
    cv2.imshow('Orig kinda', masked)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break





#  args = [None, cv2.COLOR_BGR2HSV,]
#
#  for arg in args:
#      if arg:
#          converted = cv2.cvtColor(im, arg)
#      else:
#          converted = im
#
#      ch0 = converted[:, :, 0]
#      ch1 = converted[:, :, 1]
#      ch2 = converted[:, :, 2]
#
#      cv2.imshow('Channel 0', ch0)
#      cv2.imshow('Channel 1', ch1)
#      cv2.imshow('Channel 2', ch2)
#
#      cv2.waitKey(0)
