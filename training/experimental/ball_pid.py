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

import argparse
import numpy as np
import cv2
import time

from object_detector import ObjectDetector as TFObjectDetector

parser = argparse.ArgumentParser()
parser.add_argument("--path_to_model", type=str,
        default="output_inference_graph/frozen_inference_graph.pb",
        help="Directory containing frozen checkpoint file or .tflite model")
parser.add_argument("--path_to_labels", type=str,
        default="train_data/label.pbtxt",
        help="Text proto (TF) or text (tflite) file containing label map")
parser.add_argument("--num_classes", type=int, default=2,
        help="Number of classes")
args = parser.parse_args()

cam = cv2.VideoCapture(0)
objdet = TFObjectDetector(args.path_to_model, args.path_to_labels,
        args.num_classes)

while True:

    ret, frame = cam.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    expand = np.expand_dims(img, axis=0)
    result = objdet.detect(expand)
    # detection_boxes, num_detections, detection_scores, detection_classes

    # Detection scores are sorted in descending order, so find the first one
    # which has a class of 1.
    box = None
    for i, score in enumerate(result['detection_scores']):
        if score > 0.6 and result['detection_classes'][i] == 1:

            box = result['detection_boxes'][i]
            y1, x1 = int(box[0] * h), int(box[1] * w)
            y2, x2 = int(box[2] * h), int(box[3] * w)
        
            w1 = x2-x1
            h1 = y2-y1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)

            break

    print(box)
    if box is not None:
        # Determine center of box

        #  y1n, x1n, y2n, x2n = box # Unpack it
        ycn, xcn = box[:2] + (box[2:] - box[:2]) / 2 # Given as [ycn, xcn]
        xc = int(xcn * w)
        yc = int(ycn * h)

        cv2.circle(frame, (xc, yc), 5, (255, 0, 0))

        # Based on where the center is, compute an error metric.
        target_x = 0.5
        
        # Positive error means the ball is to the right
        # Normalized coordinates in image space, so error is in [-0.5, 0.5]
        error = xcn - target_x

        Kp = 1.5 

        pid_output = Kp * error

        left_command = pid_output
        right_command = -pid_output
        print(left_command, right_command)

        draw_left_x = int(0.4 * w)
        draw_right_x = int(0.6 * w)
        cv2.arrowedLine(frame, (draw_left_x, h//2), 
                               (draw_left_x, int(h//2 * (1 - left_command))),
                               (0, 255, 255))
        cv2.arrowedLine(frame, (draw_right_x, h//2), 
                               (draw_right_x, int(h//2 * (1 - right_command))),
                               (0, 255, 255))



    cv2.imshow("pid", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
