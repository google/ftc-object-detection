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

import time
import numpy as np
import os
import errno
import sys
from object_detector import ObjectDetector
import cv2

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--movie", type=str, default="movie.mp4",
        help="Movie file to run prediction on")
parser.add_argument("--path_to_ckpt", type=str,
        default="output_inference_graph/frozen_inference_graph.pb",
        help="Directory containing frozen checkpoint file")
parser.add_argument("--path_to_labels", type=str,
        default="train_data/label.pbtxt",
        help="Text proto file containing label map")
parser.add_argument("--num_classes", type=int, default=2,
        help="Number of classes")
args = parser.parse_args()

from layers import layers

def get_blur_experiment_results(video, blur):
    cam = cv2.VideoCapture(args.movie)
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    objdet = ObjectDetector(args.path_to_ckpt, args.path_to_labels,
            args.num_classes)

    detection_counts = [] # Just the raw number of detections
    weighted_detection_counts = [] # Sum the scores

    ret, frame = cam.read()
    while ret == True:
        img = frame # Aliased, but lets us turn off as necessary.

        img = cv2.GaussianBlur(img, (blur, blur), 0) # Magic happens here

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        expand = np.expand_dims(img, axis=0)
        result = objdet.detect(expand)
        boxes = []

        detection_counts.append(0)
        weighted_detection_counts.append(0)

        for i in range(result['num_detections']):
            if result['detection_scores'][i] > 0.6:
                class_ = result['detection_classes'][i]
                box = result['detection_boxes'][i]
                score = result['detection_scores'][i]
                y1, x1 = int(box[0] * h), int(box[1] * w)
                y2, x2 = int(box[2] * h), int(box[3] * w)
                boxes.append((class_, score, x1, y1, x2, y2))

                # Less efficient, but keeps it all in the same place.
                weighted_detection_counts[-1] += score
                detection_counts[-1] += 1

        for box in boxes:
            class_, score, x1, y1, x2, y2 = box
            w1 = x2-x1
            h1 = y2-y1
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.putText(img, "%s: %5.2f" % (layers[class_-1], score),
                    (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ret, frame = cam.read()


    return np.array(detection_counts), np.array(weighted_detection_counts)


def main():

    if not os.path.exists(args.movie):
        print("Movie file %s missing" % args.movie)
        sys.exit(1)

    blurs = list(range(1, 152, 10))

    detections = []
    weighted = []

    for blur in blurs:
        print("Running experiment with blur (%d, %d)" % (blur, blur))
        result = get_blur_experiment_results(args.movie, blur)
        detections.append(result[0])
        weighted.append(result[1])

        print("Detections:", detections)
        print("Weighted:", weighted)

        np.savez("blur_experiment.npz", blurs=blurs, detections=detections,
                weighted=weighted)


if __name__ == "__main__":
    main()
