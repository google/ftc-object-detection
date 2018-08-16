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

import cv2
import numpy as np
import time

# Create the window
window_name = 'Mask'
cv2.namedWindow(window_name)

# Create the trackbars
cv2.createTrackbar('min_1', window_name, 0, 255, lambda x: None)
cv2.createTrackbar('max_1', window_name, 0, 255, lambda x: None)
cv2.createTrackbar('min_2', window_name, 0, 255, lambda x: None)
cv2.createTrackbar('max_2', window_name, 0, 255, lambda x: None)
cv2.createTrackbar('min_3', window_name, 0, 255, lambda x: None)
cv2.createTrackbar('max_3', window_name, 0, 255, lambda x: None)

vid = cv2.VideoCapture('vid1.mp4')

count = 0
while vid.isOpened():
    ret, frame = vid.read()
    count += 1

    if count < 500:
        continue

    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    blurred = cv2.blur(frame, (11, 11))
    converted = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    #  cv2.imshow('frame', frame)
    #  cv2.imshow('cvt', converted)
    #  cv2.waitKey(0)
    #  break

    # Get the channels
    ch0 = converted[:, :, 0]
    ch1 = converted[:, :, 1]
    ch2 = converted[:, :, 2]

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
    masked = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow(window_name, mask)
    cv2.imshow('Orig masked', masked)
    cv2.imshow('Original', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
