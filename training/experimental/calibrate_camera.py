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

"""Script used for camera calibration.

Currently, this will look for a checkerboard pattern (9x6) and read images from
the webcam. Every INTER_FRAME_TIME (3) seconds, until MAX_FRAMES (30) have been
collected, the camera will take a picture. You should move the checkerboard
around the full field of view of the camera, making sure to rotate the
checkerboard through all different orientations and rotations in XYZ.

Once MAX_FRAMES pictures have been gathered, RANSAC is used to determine the
best approximation for the camera calibration. Finally, the calibration is saved
to file (calib.npz), and the calibrated video stream is displayed on screen.

You can get a checkerboard pattern from the link below. Note that you'll
potentially need to adjust square_size.

https://www.mrpt.org/downloads/camera-calibration-checker-board_9x7.pdf
"""

import cv2
import numpy as np
import sys
import argparse
import collections
import time
import os


# Results from moto:

parser = argparse.ArgumentParser()
input_type = parser.add_mutually_exclusive_group(required=True)
input_type.add_argument("--video", type=str, default="",
        help="Video to read calibration frames from.")
input_type.add_argument("--camera", action="store_true",
        help="Use the camera to collect frames")
input_type.add_argument("--images", type=str, default="",
        help="Directory for calibration images")
args = parser.parse_args()


def get_points_from_frame(frame):

    # Calibration stuff
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    square_size = 25 # mm
    pattern_size=(9, 6)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    print("Finding chessboard corners")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        #  cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(frame, pattern_size, corners, ret)
        cv2.imshow('Checkerboard', frame)
        cv2.waitKey(1)

        img_points = corners.reshape(-1, 2)
        obj_points = pattern_points

        return img_points, obj_points


def get_calib_inputs_from_folder(folder):

    obj_points = []
    img_points = []

    width = None
    height = None

    # Read images from this folder
    for f in os.listdir(folder):
        full_path = os.path.join(folder, f)
        if os.path.isfile(full_path):
            frame = cv2.imread(full_path)
            height, width, _ = frame.shape

            ret = get_points_from_frame(frame)
            if ret:
                img_points.append(ret[0])
                obj_points.append(ret[1])

    return img_points, obj_points, width, height


def get_calib_inputs_from_cap(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(width, height, fps)

    obj_points = []
    img_points = []

    INTER_FRAME_TIME = 3 # seconds
    INTER_FRAME_FRAMES = INTER_FRAME_TIME * fps

    elapsed_frames = 0
    last_elapsed_frames = -INTER_FRAME_FRAMES
    MAX_FRAMES = 30

    while len(img_points) < MAX_FRAMES:

        ret, frame = cap.read()
        elapsed_frames += 1
        if not ret:
            print("Ran out of frames, exiting!")
            sys.exit()

        cv2.imshow("Preview", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Ordered to exit, stopping.")
            sys.exit()

        if elapsed_frames - last_elapsed_frames < INTER_FRAME_FRAMES:
            continue

        last_elapsed_frames = elapsed_frames

        ret = get_points_from_frame(frame)
        if ret:
            print("Adding a new frame.")
            img_points.append(ret[0])
            obj_points.append(ret[1])

    cap.close()
    return img_points, obj_points, width, height


def ransac(img_points, obj_points, width, height):
    # Do RANSAC to figure out the best set
    print("Performing RANSAC")
    best_error = 1000 # or some other high number
    other_stuff = None

    for i in range(100):
        indices = np.random.randint(len(img_points), size=10)
        # Can use this to disable RANSAC
        #  indices = np.random.choice(range(len(img_points)), size=len(img_points),
        #          replace=False)
        sampled_img_points = [img_points[i] for i in indices]
        sampled_obj_points = [obj_points[i] for i in indices]

        rms, *other = cv2.calibrateCamera(
                sampled_obj_points, sampled_img_points,
                (width, height), None, None)

        if rms < best_error:
            best_error = rms
            other_stuff = other

    print("Final RANSAC RMS Error:", best_error)
    camera_matrix, dist_coefs, rvecs, tvecs = other_stuff

    np.savez("calib.npz", camera_matrix=camera_matrix, dist_coefs=dist_coefs)

    print("RMS:", rms)
    print("camera_matrix:\n", camera_matrix)
    print("distortion coefs:", dist_coefs.ravel())

    return camera_matrix, dist_coefs


def show_calibrated_video(camera_matrix, dist_coefs):

    if args.video:
        cap = cv2.VideoCapture(args.video)
    elif args.camera:
        cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coefs, (width, height), 1, (width, height))
        dst = cv2.undistort(frame, camera_matrix, dist_coefs, None, newcameramtx)
        cv2.imshow("warped", dst)
        cv2.imshow("orig", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return


def show_calibrated_images(camera_matrix, dist_coefs):

    while True:
        # Read images from this folder
        for f in os.listdir(args.images):
            full_path = os.path.join(args.images, f)
            if os.path.isfile(full_path):
                frame = cv2.imread(full_path)
                height, width, _ = frame.shape

                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                        camera_matrix, dist_coefs, (width, height),
                        1, (width, height))
                dst = cv2.undistort(frame, camera_matrix, dist_coefs,
                        None, newcameramtx)
                cv2.imshow("warped", dst)
                cv2.imshow("orig", frame)

                if cv2.waitKey(1000) & 0xFF == ord('q'):
                    return


def show_calibrated(camera_matrix, dist_coefs):
    if args.images:
        show_calibrated_images(camera_matrix, dist_coefs)
    else:
        show_calibrated_video(camera_matrix, dist_coefs)


if __name__ == "__main__":
    if args.video:
        cap = cv2.VideoCapture(args.video)
        calib_inputs = get_calib_inputs_from_cap(cap)
    elif args.camera:
        cap = cv2.VideoCapture(0)
        calib_inputs = get_calib_inputs_from_cap(cap)
    elif args.images:
        calib_inputs = get_calib_inputs_from_folder(args.images)

    calib_data = ransac(*calib_inputs)

    show_calibrated(*calib_data)
    cv2.destroyAllWindows()
