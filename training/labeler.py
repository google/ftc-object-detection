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
import numpy as np
import argparse
import drawing_utils
import os
import bbox_writer


WINDOW = "Tracking"
WINDOW_SCALE = .75
CACHE_SIZE = 150 # 5 seconds worth of frames

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=argparse.FileType('r'))
parser.add_argument("-f", "--frames", type=int,
        help="Number of steps between each frame to save.", default=10)
args = parser.parse_args()

last_bboxes = []
last_classes = []


def open_vid(path):
    # Open the video
    vid = cv2.VideoCapture(path)
    if not vid.isOpened():
        print("Unable to open video")
        sys.exit()
    return vid


def show_scaled(window, frame, sf=WINDOW_SCALE):
    cv2.imshow(window, cv2.resize(frame, (0, 0), fx=sf, fy=sf))


def draw_text(image, text, location):
    font_color = (170, 50, 50)
    font_weight = 2
    font_scale = 0.75
    font_type = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(image, text, (location[0] + 2, location[1] + 2), font_type,
            font_scale, (255, 255, 255), font_weight)

    cv2.putText(image, text, location, font_type, font_scale, font_color,
            font_weight)


def draw_frame_text(frame, frame_text):
    for i, line in enumerate(frame_text):
        draw_text(frame, line, (20, i * 30 + 30))


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


# Let the user do some labeling. If they press any key that doesn't map to a
# useful thing here, we return it.
def label_frame(original, bboxes, classes, frame_text):
    global last_bboxes, last_classes

    points = []
    current_class = None
    shift_pressed = False

    def draw(frame):
        drawing_utils.draw_bboxes(frame, bboxes, classes)
        draw_frame_text(frame, frame_text + ["Current class: %s" %
            current_class])
        show_scaled(WINDOW, frame)


    def mouse_callback(event, x, y, flags, params):
        frame = params.copy() # Copy of original that we can afford to draw on
        h, w, c = frame.shape

        # (x, y) in original image coordinates
        x = int(x / WINDOW_SCALE)
        y = int(y / WINDOW_SCALE)

        if event == cv2.EVENT_LBUTTONDOWN:
            points.append(np.array([x, y]))

        if len(points) == 1: # Still drawing a rectangle
            cv2.rectangle(frame, tuple(points[0]), (x, y), (255, 255, 0), 1, 1)

        # If the mouse is moved, draw crosshairs
        cv2.line(frame, (x, 0), (x, h), (255, 0, 0))
        cv2.line(frame, (0, y), (w, y), (255, 0, 0))

        if len(points) == 2: # We've got a rectangle
            bbox = np.array([points[0], points[1] - points[0]]).reshape(-1)
            bbox = standardize_bbox(bbox)

            cls = str(current_class)
            bboxes.append(bbox)
            classes.append(cls)
            points.clear()

        draw(frame)


    cv2.setMouseCallback(WINDOW, mouse_callback, param=original)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == 225 or key == 226: # shift, seems to be platform dependent?
            shift_pressed = True

        elif shift_pressed and ord('a') <= key <= ord('z'):
            current_class = chr(key)
            shift_pressed = False
            draw(original.copy())

        elif key == ord('c'): # Clear everything
            bboxes.clear()
            classes.clear()
            draw(original.copy())

        elif key == ord('x'): # Remove most recently placed box
            if len(bboxes) > 0:
                bboxes.pop(len(bboxes) - 1)
                classes.pop(len(classes) - 1)
                draw(original.copy())

        elif key == ord('r'): # Get the data from the last label session
            bboxes.clear()
            classes.clear()

            bboxes.extend(last_bboxes)
            classes.extend(last_classes)
            draw(original.copy())

        elif key != 255: # Default return value from waitKey, keep labeling
            break


    # Only save if we have non-empty labels
    if bboxes:
        last_bboxes = bboxes

    if classes:
        last_classes = classes

    cv2.setMouseCallback(WINDOW, lambda *args: None)
    return key


def load_bboxes(frame_number, run_path):
    # Figure out which file we're trying to load. First, get the path of the
    # image file that we'd be saving against.
    bbox_filename = os.path.join(run_path, "%05d.txt" % frame_number)
    if os.path.isfile(bbox_filename):
        bboxes, classes = bbox_writer.read_bboxes(bbox_filename)
    else:
        # Not saved yet, so just return an empty list
        bboxes = []
        classes = []

    return bboxes, classes


def save_frame(frame, bboxes, classes, run_path, frame_number):

    frame_path = os.path.join(run_path, "%05d.png" % frame_number)
    bbox_path = os.path.join(run_path, "%05d.txt" % frame_number)

    if not os.path.isfile(frame_path):
        print("Saving frame %d to %s" % (frame_number, frame_path))
        cv2.imwrite(frame_path, frame)
    bbox_writer.write_bboxes(bboxes, classes, bbox_path)


def main():
    vid = open_vid(args.filename.name)

    autoplay = False
    autoplay_delay = 32
    stop_at_next_save = False

    current_frame_number = 0
    last_removed_frame = -1

    stored_frames = dict()

    # Initialize the storage on disk
    filename = os.path.splitext(os.path.basename(args.filename.name))[0]
    run_name = "%s" % (filename)
    run_path = os.path.join(os.path.dirname(args.filename.name), run_name)

    try:
        os.mkdir(run_path)
    except:
        print("Directory probably exists already, continuing anyway.")

    while True:

        is_save_frame = (args.frames > 0 and
                current_frame_number % args.frames == 0)

        if current_frame_number not in stored_frames:
            ret, frame = vid.read()

            if not ret:
                print("Unable to open frame, quitting!")
                break

            # If this is a frame we care about, save it to disk. Also, see if
            # there is already a saved set of bboxes, and load those if they
            # exist.
            bboxes, classes = load_bboxes(current_frame_number, run_path)

            stored_frames[current_frame_number] = (frame, bboxes, classes)
            if len(stored_frames) > CACHE_SIZE:
                last_removed_frame += 1
                print("Removing frame", last_removed_frame, " from cache")
                stored_frames.pop(last_removed_frame)

        else:
            frame, bboxes, classes = stored_frames[current_frame_number]


        drawable_frame = frame.copy()
        frame_text = [
            "Frame number: " + str(current_frame_number) +
            (" (saved)" if is_save_frame else ""),
            "Autoplay: " + str(autoplay),
            "Autoplay delay: " + str(autoplay_delay),
            "Stopping at next save frame: " + str(stop_at_next_save),
        ]
        draw_frame_text(drawable_frame, frame_text)
        drawing_utils.draw_bboxes(drawable_frame, bboxes, classes)

        show_scaled(WINDOW, drawable_frame)

        if autoplay:
            delay = autoplay_delay if autoplay else 0
            key = cv2.waitKey(delay) & 0xFF
        else:
            key = label_frame(frame, bboxes, classes, frame_text)

        if is_save_frame:
            save_frame(frame, bboxes, classes, run_path, current_frame_number)

            if stop_at_next_save:
                stop_at_next_save = False
                autoplay = False


        # Handle whatever key the user pressed. The user may have potentially
        # labeled something, as above.
        if key == ord('q'):
            break
        if key == ord('l'):
            current_frame_number += 1
        elif key == ord('h'):
            current_frame_number -= 1
            current_frame_number = max(current_frame_number,
                    last_removed_frame + 1)
        elif key == ord('j'):
            autoplay_delay = max(autoplay_delay // 2, 1)
        elif key == ord('k'):
            autoplay_delay *= 2
        elif key == ord(' '):
            autoplay = not autoplay
            autoplay_delay = 32
        elif key == ord('n'):
            stop_at_next_save = True
            autoplay = True
            autoplay_delay = 1
            current_frame_number += 1
        elif autoplay:
            current_frame_number += 1

    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
