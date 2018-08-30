#!/usr/bin/env python3

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
import sys
import os
import numpy as np
import bbox_writer
import argparse
import drawing_utils

description_text="""\
Use this script to draw initial bounding boxes on videos.

This script is meant to work hand in hand with tracking.py. To label a new
video, you should first annotate bounding boxes on the video's first frame with
this script, and then load the video into tracking.py to track the bounding
boxes.

To label an object, you should first pick the class that the object belongs to,
by pressing any (lowercase) alphanumeric character on your keyboard. This will
set the class to be that single character, e.g. 'v', 'a', 's', 'u', etc. After
picking a class, you can use your mouse to click on two opposing corners of a
box around an object. After you click the second point, the object will be
labeled with the class you specified earlier. At this point, you can change the
class, or draw another box with the same class.

Once you're satisfied with the boxes you've drawn, you can exit the program by
pressing 'q'. This also means that 'q' is not a valid object class.

If you would like to clear the first point you selected, you can press Escape.
If you would like to remove a bounding box you've drawn already, you should
simply restart the program, which will clear all bounding boxes.

Finally, you should try to make bounding boxes drawn here as tight as possible.
Any scaling of bounding boxes that needs to be performed can be done in
tracking.py.
"""

parser = argparse.ArgumentParser(
        description=description_text,
        formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("filename", type=argparse.FileType("r"),
        help="path to video file")
args = parser.parse_args()

window = 'draw on me!'
points = []
bboxes = []
classes = []
current_class = None

WINDOW_SCALE = 0.5

def show_scaled(window, frame, sf=WINDOW_SCALE):
    cv2.imshow(window, cv2.resize(frame, (0, 0), fx=sf, fy=sf))

# Make sure all bboxes are given as top left (x, y), and (dx, dy). Sometimes
# they may be specified by a different corner, so we need to reverse that.
def standardize_bbox(bbox):
    p0 = bbox[:2]
    p1 = p0 + bbox[2:]

    min_x = min(p0[0], p1[0])
    max_x = max(p0[0], p1[0])
    min_y = min(p0[1], p1[1])
    max_y = max(p0[1], p1[1])

    ret = np.array([min_x, min_y, max_x - min_x, max_y - min_y])
    print("Standardized %s to %s" % (bbox, ret))

    return ret


def mouse_callback(event, x, y, flags, params):
    im = params.copy()
    h, w, c = im.shape

    x = int(x / WINDOW_SCALE)
    y = int(y / WINDOW_SCALE)

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append(np.array([x, y]))

    if len(points) == 1: # Still drawing a rectangle
        cv2.rectangle(im, tuple(points[0]), (x, y), (255, 255, 0), 1, 1)

    # If the mouse is moved, draw crosshairs
    cv2.line(im, (x, 0), (x, h), (255, 0, 0))
    cv2.line(im, (0, y), (w, y), (255, 0, 0))

    if len(points) == 2: # We've got a rectangle
        bbox = np.array([points[0], points[1] - points[0]]).reshape(-1)
        bbox = standardize_bbox(bbox)

        cls = str(current_class)
        bboxes.append(bbox)
        classes.append(cls)
        points.clear()

        # Write the bboxes out to file.
        # We get the filename itself, not the full path.
        filename = args.filename.name
        bbox_filename = bbox_writer.get_bbox_filename(filename)
        bbox_path = os.path.join(os.path.dirname(filename), bbox_filename)
        bbox_writer.write_bboxes(bboxes, classes, bbox_path)
        print("Wrote bboxes to file:", bbox_path)

    drawing_utils.draw_bboxes(im, bboxes, classes)

    cv2.putText(im, "Current class: %s" % current_class, (100,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    show_scaled(window, im)


if __name__ == "__main__":

    vid = cv2.VideoCapture(args.filename.name)
    cv2.namedWindow(window)

    if vid.isOpened():
        ret, frame = vid.read()
        show_scaled(window, frame)
        cv2.setMouseCallback(window, mouse_callback, param=frame)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 27: # esc
            points.clear()
        elif key > -1 and key < 128 and chr(key).isalnum():
            current_class = chr(key)
