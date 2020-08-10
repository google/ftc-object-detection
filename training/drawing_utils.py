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


def scale_bboxes(bboxes, sf):

    scaled_bboxes = []

    for bbox in bboxes:
        if bbox is None:
            scaled_bboxes.append(None)
        else:
            p0 = bbox[:2].astype(float)
            p1 = p0 + bbox[2:].astype(float)
            size = p1 - p0
            center = p0 + (size / 2)

            new_size = sf * size
            p0 = center - new_size / 2
            p1 = center + new_size / 2

            scaled_bboxes.append(np.array([p0, p1 - p0]).reshape(-1))

    return scaled_bboxes


def draw_bboxes(frame, bboxes, classes, scale=1):
    assert(len(bboxes) == len(classes))

    scaled_bboxes = scale_bboxes(bboxes, 1 / scale)

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        cls = classes[i]

        if bbox is None or cls is None:
            continue

        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 255), 2, 1)
        cv2.putText(frame, cls, (int(bbox[0]), int(bbox[1] + bbox[3] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)


        if scale != 1:
            bbox = scaled_bboxes[i]
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 100), 2, 1)


def draw_dots(frame, bboxes):
    # Draw dots on the top left and bottom right corners of bboxes
    node_color = (255, 255, 255)
    for bbox in bboxes:
        if bbox is not None:
            p0 = bbox[:2]
            p1 = p0 + bbox[2:]
            cv2.circle(frame, tuple(p0.astype(int)), 10, node_color,
                    thickness=-1)
            cv2.circle(frame, tuple(p1.astype(int)), 10, node_color,
                    thickness=-1)
