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
import sys
import time
import pdb
import os
import bbox_writer
import multiprocessing
import argparse
import drawing_utils

description_text="""\
Use this script to generate labeled image pairs from an initial labeled frame.

This script is meant to work hand in hand with find_bb.py. To label a new video,
you should first annotate bounding boxes with find_bb.py, and then load the
video into this script to track those bounding boxes.

By default, you will first be asked to verify the loaded annotations for the
video. You should sanity check these annotations to make sure that the bounding
boxes are tight, and that the classes are correct. Press 'y' to indicate
correctness, or 'n' to indicate that something is wrong. If you press 'n', you
should correct the labels by running find_bb.py again.

Once you press 'y', the tracking will start. Each bounding box will be tracked
through each frame with the specified tracker. You may see two bounding boxes, a
green and a yellow one. The yellow one is affected by the --scale parameter, and
is the region the tracker is tracking. The green one is the one that actually
gets saved. If you only see a yellow box, the two boxes are identical.

If at any point the tracked boxes deviate too far from the object, you can pause
execution by pressing SpaceBar. You can then click and drag the white circles to
correct the bounding boxes. Remember that you'll usually want the green box to
be as tight as possible to the object of interest. Once you are satisfied with
the new boxes, press SpaceBar again to continue.

The script will automatically terminate once the end of the video is reached.
Alternatively, you can press 'q' to terminate early.
"""

epilog_text="""\
example:
    ./tracking.py [filename].mp4                    use all default paramters
    ./tracking.py [filename].mp4 -s 1.2             change scale for tracker box
    ./tracking.py [filename].mp4 -x -s 1.5 -t 1     try out different parameters
"""

parser = argparse.ArgumentParser(
        description=description_text,
        epilog=epilog_text,
        formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("filename", type=argparse.FileType('r'))
parser.add_argument("-s", "--scale", type=float, default=1.0, required=False,
        help="Scale factor to help the tracker along")
parser.add_argument("-t", "--tracker", type=int, default=2, required=False,
        help="Index of tracker to use, [0-7]")
parser.add_argument("-y", "--yes", action="store_true", default=False,
        help="Skip initial bounding box validation")
parser.add_argument("-f", "--frames", type=int,
        help="Number of steps between each frame to save", default=10)
parser.add_argument("-x", "--experiment", action="store_true", default=False,
        help="Don't write out any files")
parser.add_argument("-r", "--refine", action="store_true", default=False,
        help="Auto-refine bounding boxes during tracking (experimental)")
args = parser.parse_args()

window = "Tracking"
WINDOW_SCALE = 0.5

# Make a bunch of trackers because I don't trust the opencv one
tracker_fns = [
        cv2.TrackerKCF_create,
        cv2.TrackerBoosting_create,
        cv2.TrackerCSRT_create,
        cv2.TrackerGOTURN_create,
        cv2.TrackerMIL_create,
        cv2.TrackerMOSSE_create,
        cv2.TrackerMedianFlow_create,
        cv2.TrackerTLD_create,
]


def show_scaled(window, frame, sf=WINDOW_SCALE):
    cv2.imshow(window, cv2.resize(frame, (0, 0), fx=sf, fy=sf))


def get_scaled_bboxes(filename, sf):
    bbox_filename = bbox_writer.get_bbox_filename(filename)
    bbox_path = os.path.join(os.path.dirname(filename), bbox_filename)
    bboxes_, classes = bbox_writer.read_bboxes(bbox_path)
    bboxes = drawing_utils.scale_bboxes(bboxes_, sf)

    return bboxes, classes


def open_vid(path):
    # Open the video
    vid = cv2.VideoCapture(path)
    if not vid.isOpened():
        print("Unable to open video")
        sys.exit()
    return vid


def verify_bboxes(frame, bboxes, classes, yes):

    frame_to_draw = frame.copy()
    drawing_utils.draw_bboxes(frame_to_draw, bboxes, classes, args.scale)
    cv2.putText(frame_to_draw, "Do boxes look okay (y/n)?", (100, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
    show_scaled(window, frame_to_draw)

    if yes:
       return

    # Confirm sanity check
    key = cv2.waitKey(0) & 0xFF
    if key == ord('n'):
        print("Poor bounding boxes. Quitting!")
        sys.exit()


def init_trackers(tracker_index, frame, bboxes):
    trackers = []
    tracker_fn = tracker_fns[tracker_index]

    for i, bbox in enumerate(bboxes):
        tracker = tracker_fn()
        ret = tracker.init(frame, tuple(bbox))
        if not ret:
            print("Unable to initialize tracker", i)
            continue
        else:
            print("Successfully initialized tracker", i)
            trackers.append(tracker)

    return trackers


def save_frame(orig, frame, bboxes, classes, run_name, frame_count):

    # Scale the bboxes back down by original scale factor.
    # This gives us the tight bounding box for the object, rather than the one
    # which has been scaled for the tracker.
    bboxes = drawing_utils.scale_bboxes(bboxes, 1 / args.scale)

    frame_to_draw = orig.copy()
    drawing_utils.draw_bboxes(frame_to_draw, bboxes, classes)
    show_scaled("Saved Frame", frame_to_draw)

    if args.experiment:
        return

    cv2.imwrite(os.path.join(run_name, "%05d.png" % frame_count), orig)
    cv2.imwrite(os.path.join(run_name, "rect_%05d.png" % frame_count), frame)
    bbox_writer.write_bboxes(bboxes, classes,
            os.path.join(run_name, "%05d.txt" % frame_count))



def correction_mode(orig, trackers, bboxes, classes, annotated_classes):

    frame = orig.copy()
    drawing_utils.draw_bboxes(frame, bboxes, annotated_classes, args.scale)
    drawing_utils.draw_dots(frame, bboxes)

    show_scaled(window, frame)

    modified = set()
    tracked_point = None
    is_top_left = False

    def mouse_callback(event, x, y, flags, params):
        nonlocal tracked_point, is_top_left

        orig, trackers, bboxes, classes = params
        im = orig.copy()

        # Determine which bbox is corresponded to by the click
        click = np.array([x, y]) / WINDOW_SCALE
        radius = 10

        # If there is no tracked point, determine which point gets clicked, if
        # any, and set variables accordingly.
        if tracked_point is None and event == cv2.EVENT_LBUTTONDOWN:
            for i, bbox in enumerate(bboxes):
                p0 = bbox[:2]
                p1 = p0 + bbox[2:]

                if np.linalg.norm(p0 - click) < radius:
                    tracked_point = i
                    modified.add(tracked_point)
                    is_top_left = True
                    break
                elif np.linalg.norm(p1 - click) < radius:
                    tracked_point = i
                    modified.add(tracked_point)
                    is_top_left = False
                    break
        elif tracked_point is not None and event == cv2.EVENT_LBUTTONDOWN:
            tracked_point = None
        elif tracked_point is not None:
            # There must be a tracked point, so move the point to the location
            # of the mouse click.
            p0 = bboxes[tracked_point][:2]
            p1 = p0 + bboxes[tracked_point][2:]
            if is_top_left: # No fancy handling necessary
                bboxes[tracked_point][:2] = click
                bboxes[tracked_point][2:] = p1 - click
            else:
                bboxes[tracked_point][2:] = click - bboxes[tracked_point][:2]

        drawing_utils.draw_bboxes(im, bboxes, classes, args.scale)
        drawing_utils.draw_dots(im, bboxes)
        show_scaled(window, im)

    cv2.setMouseCallback(window, mouse_callback,
            param=(orig, trackers, bboxes, annotated_classes))

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') or key == ord(' '):
            for mod in modified:
                print("Reinitializing tracker %d" % mod)
                new_tracker = init_trackers(args.tracker, orig, [bboxes[mod]])
                trackers[mod] = new_tracker[0]
            break
        elif key == ord('r'):
            refine_bboxes(bboxes, classes, orig, trackers)

    # Clear the mouse callback
    cv2.setMouseCallback(window, lambda *args: None)


def clamp_bboxes(bboxes, width, height):

    clamped_bboxes = []
    for bbox in bboxes:
        p1 = bbox[:2]
        p2 = p1 + bbox[2:]

        p1[0] = np.clip(p1[0], 0, width)
        p1[1] = np.clip(p1[1], 0, height)

        p2[0] = np.clip(p2[0], 0, width)
        p2[1] = np.clip(p2[1], 0, height)

        clamped_bbox = np.array([*p1, *(p2 - p1)])
        clamped_bboxes.append(clamped_bbox)

    return clamped_bboxes


def refine_bboxes(bboxes, classes, frame, trackers):
    # Refine boxes and reinitialize trackers.
    # Boxes are refined to be as tight as possible to the object being tracked.
    # The tracker is then given the bbox which has been inflated by the original
    # scale factor, to preserve tracking quality.

    # Just in case the tracker is missing something, we scale even further to
    # determine our ROI.
    scaled_bboxes = drawing_utils.scale_bboxes(bboxes, 1.2)

    h, w, _ = frame.shape

    # Very much hard coded for our particular use case.
    for i, bbox in enumerate(scaled_bboxes):
        if bbox is None: continue

        # Grab the part that we care about.
        rounded_bbox = bbox.astype(int)
        top_left = rounded_bbox[:2]
        bottom_right = top_left + rounded_bbox[2:]
        xs = np.clip([top_left[0], bottom_right[0]], 0, w)
        ys = np.clip([top_left[1], bottom_right[1]], 0, h)

        roi = frame[ys[0]:ys[1], xs[0]:xs[1]]

        # Resize the roi to be a reasonable dimension to see
        # Make the smaller of the two dimensions a fixed size
        IMAGE_SIZE = 100
        roi_h, roi_w, _ = roi.shape
        sf = IMAGE_SIZE / min(roi_h, roi_w)
        roi = cv2.resize(roi, (0, 0), fx=sf, fy=sf)

        new_bbox = None
        cls = classes[i]
        if cls == 'w':
            # TODO: Tune parameters here, if necessary
            print("Refining white whiffle ball")
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            min_radius = IMAGE_SIZE // 4
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1,
                    minDist=IMAGE_SIZE/2, param1=30, param2=50,
                    minRadius=min_radius,
                    maxRadius=IMAGE_SIZE//2)
            if circles is None:
                print("NO CIRCLES DETECTED. UHHHH")
                continue

            # Find the biggest circle by area, aka biggest radius
            biggest_circle_index = np.argmax(circles[0, :, 2])
            biggest_circle = circles[0, biggest_circle_index]
            c = biggest_circle

            if (c[2] < min_radius):
                print("Got an invalid circle?")
                continue

            # draw the outer circle and a dot at the center
            cv2.circle(roi, (c[0], c[1]), c[2], (0, 255, 0), 2)
            cv2.circle(roi, (c[0], c[1]), 2, (0, 0, 255), 3)

            # Use the bounding box of the circle to reinitialize the tracker.
            new_bbox = np.array([c[0] - c[2], c[1] - c[2], 2 * c[2], 2 * c[2]])


        elif cls == 'c':
            print("Refining orange cube")
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hsv_blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
            ret, thresh_h = cv2.threshold(hsv_blurred[:, :, 0], 30, 255,
                    cv2.THRESH_BINARY_INV)
            ret, thresh_s = cv2.threshold(hsv_blurred[:, :, 1], 0, 255,
                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            mask = cv2.bitwise_and(thresh_h, thresh_s)


            # Clean up the mask a little
            kernel = np.ones((11,11),np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            #  cv2.imshow("Opening", opening)


            roi = cv2.bitwise_and(roi, roi, mask=mask)
            print("made the roi from the mask")

            # Grab the bounding box from the mask
            conn_stats = cv2.connectedComponentsWithStats(mask, connectivity=4)
            retval, labels, stats, centroids = conn_stats

            # The stats tell us [top left, top right, width, height, area]
            # Find the label with the biggest area
            if len(stats) > 1: # Means we have a non-bg label
                biggest_label = np.argmax(stats[1:, -1]) + 1

                p1 = stats[biggest_label, :2]
                p2 = p1 + stats[biggest_label, 2:-1]
                cv2.rectangle(roi, tuple(p1.astype(int)), tuple(p2.astype(int)), color=(255, 0, 100))
                print("drew the rectangle")

                new_bbox = stats[biggest_label, :-1]

        cv2.imshow("Image %d" % i, roi)

        if new_bbox is None:
            continue

        print("New bounding box", new_bbox)
        new_bbox = new_bbox / sf # Unscale by the same amount we scaled
        new_bbox = np.array([*(top_left + new_bbox[:2]), *new_bbox[2:]])

        print("Replacing bbox %d" % i, rounded_bbox, new_bbox)

        # Scale the bbox by the proper scale factor
        new_bbox_scaled = drawing_utils.scale_bboxes([new_bbox], args.scale)
        new_bbox_scaled = clamp_bboxes(new_bbox_scaled, w, h)

        # Force the scaled bounding box to be inside the bounds of the image.
        #  if any(new_bbox < 0):
        #      input()


        print("Initializing tracker")
        # Apply the new scaled bbox to both the tracker and the saved ones
        new_tracker = init_trackers(args.tracker, frame, new_bbox_scaled)[0]
        trackers[i] = new_tracker
        bboxes[i] = new_bbox_scaled[0]
        print("new scaled bbox", bboxes[i])


if __name__ == "__main__":
    bboxes, classes = get_scaled_bboxes(args.filename.name, args.scale)
    vid = open_vid(args.filename.name)

    ret, frame = vid.read()
    verify_bboxes(frame, bboxes, classes, args.yes)

    tracker_index = args.tracker
    tracker_fn = tracker_fns[tracker_index]
    tracker_name = tracker_fn.__name__.split("_")[0]

    trackers = init_trackers(tracker_index, frame, bboxes)

    # Initialize video now that we're sure we want to try to track.
    filename = os.path.splitext(os.path.basename(args.filename.name))[0]
    run_name = "%s_%s_%f" % (filename, tracker_name, args.scale)
    run_path = os.path.join(os.path.dirname(args.filename.name), run_name)

    if not args.experiment:
        try:
            os.mkdir(run_path) # Make the directory for storing images
        except:
            print("Directory probably exists already, continuing anyway.")

        writer = cv2.VideoWriter(
                    "%s.avi" % run_path,
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    int(vid.get(cv2.CAP_PROP_FPS)),
                    (frame.shape[1], frame.shape[0]),
                )

        # Quit if there was a problem
        if not writer.isOpened():
            print("Unable to open video!")
            sys.exit()

        writer.write(frame) # Write out the first image, for consistency


    frame_count = -1 # So that the second frame is saved

    # Track through each frame
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            print("Unable to open frame, quitting!")
            break

        frame_count += 1
        bboxes = []
        annotated_classes = []

        start = time.time()

        for i, tracker in enumerate(trackers):
            ret, bbox = tracker.update(frame)
            if not ret:
                print("Tracking failure for object", i)
                bboxes.append(None)
                annotated_classes.append("[FAILURE] %d:%s" % (i, classes[i]))
            else:
                bboxes.append(np.array(bbox))
                annotated_classes.append("%d:%s" % (i, classes[i]))

        if args.refine:
            refine_bboxes(bboxes, classes, frame, trackers)

        end = time.time()
        fps = 1.0 / (end - start)

        original = frame.copy()
        drawing_utils.draw_bboxes(frame, bboxes, annotated_classes, args.scale)

        # Potentially save the frame to disk using @dek's format
        if args.frames > 0 and frame_count % args.frames == 0:
            save_frame(original, frame, bboxes, classes, run_path, frame_count)

        font_color = (170, 50, 50)
        font_weight = 2
        font_scale = 0.75
        font_type = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(frame, tracker_name, (100, 20), font_type, font_scale,
                font_color, font_weight)
        cv2.putText(frame, "FPS: " + str(int(fps)), (100, 50), font_type,
                font_scale, font_color, font_weight)
        cv2.putText(frame, "Frame: " + str(frame_count), (100, 80), font_type,
                font_scale, font_color, font_weight)

        # Display result
        show_scaled(window, frame)
        if not args.experiment:
            writer.write(frame)

        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break
        elif k == ord('p') or k == ord(' '):
            # Let the user correct tracks
            correction_mode(original, trackers, bboxes, classes,
                    annotated_classes)
        elif k == ord('r'):
            refine_bboxes(bboxes, classes, original, trackers)


    cv2.waitKey(1) # Just in case

    if not args.experiment:
        writer.release()

    vid.release()
    cv2.destroyAllWindows()
