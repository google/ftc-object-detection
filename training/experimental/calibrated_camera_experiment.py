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


def vector_angle_degrees(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    angle_rad = np.arccos(np.dot(v1, v2))
    return np.rad2deg(angle_rad)


def undistort_points(distorted_points, camera_matrix, dist_coefs):
    undistorted = []

    print(camera_matrix)
    fx = camera_matrix[0][0]
    fy = camera_matrix[1][1]
    cx = camera_matrix[0][2]
    cy = camera_matrix[1][2]
    print(fx, fy, cx, cy)

    for px, py in distorted_points[0]:
        undistorted.append([(px - cx) / fx, (py - cy) / fy])

    undistorted = np.array(undistorted).reshape(1, -1, 2)

    return undistorted


loaded = np.load("calib.npz")
camera_matrix = loaded['camera_matrix']
dist_coefs = loaded['dist_coefs']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break


    # Find the orange cube
    blurred = cv2.blur(frame, (11, 11))
    converted = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    lower = np.array([13, 153, 101])
    upper = np.array([33, 255, 255])
    mask = cv2.inRange(converted, lower, upper)
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    masked = masked[:, :, 0] | masked[:, :, 1] | masked[:, :, 2]

    ret, labels = cv2.connectedComponents(masked)
    counts = np.bincount(labels.ravel())
    if len(counts) > 1:
        argmax = np.argmax(counts[1:]) + 1
        print(counts, argmax)
    else:
        continue

    biggest_blob = labels == argmax
    # https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    rows = np.any(biggest_blob, axis=1)
    cols = np.any(biggest_blob, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Draw a grid in the center, to get rough estimates for angles being correct
    image_shape = np.array([frame.shape[0], frame.shape[1]]) # 480 x 640, h x w
    cv2.line(frame, (image_shape[1] // 2, 0), (image_shape[1] // 2,
        image_shape[0] // 1), color=(0, 255, 0))
    cv2.line(frame, (0, image_shape[0] // 2), (image_shape[1] // 1,
        image_shape[0] // 2), color=(0, 255, 0))

    # Corners of the bounding box
    top_left = (cmin, rmin)
    top_right = (cmax, rmin)
    bottom_right = (cmax, rmax)
    bottom_left = (cmin, rmax)
    center = ((cmin + cmax) / 2, (rmin + rmax) / 2)
    cv2.rectangle(frame, top_left, bottom_right, color=(255, 0, 0))

    # Undistort the points. This doesn't seem to be very necessary, since most
    # phones and webcams will have very little distortion.
    distorted_points = np.array([top_left, top_right, bottom_left,
        bottom_right, center], dtype=float).reshape(1, -1, 2)
    undistorted_points = cv2.undistortPoints(distorted_points, camera_matrix,
            dist_coefs)
    custom_undistorted_points = undistort_points(distorted_points,
            camera_matrix, dist_coefs)
    print("Box points", distorted_points)
    print("Undistorted [opencv]", undistorted_points)
    print("Undistorted [normal]", custom_undistorted_points)
    print("Difference", np.linalg.norm(undistorted_points -
        custom_undistorted_points))

    undistorted_center = undistorted_points[0, -1, :]
    print("Undistorted center", undistorted_center)

    # Determine the angle to the undistorted center.
    camera_center_point = np.array([0, 0, 1])
    box_center_point = np.array([*undistorted_center, 1])
    box_center_point_x = np.array([undistorted_center[0], 0, 1])
    box_center_point_y = np.array([0, undistorted_center[1], 1])
    print("Coords", box_center_point, box_center_point_x, box_center_point_y)

    print("Angle Magnitudes")
    print("c", vector_angle_degrees(camera_center_point, box_center_point))
    print("x", vector_angle_degrees(camera_center_point, box_center_point_x))
    print("y", vector_angle_degrees(camera_center_point, box_center_point_y))

    box_x = np.array([undistorted_center[0], 1])
    box_y = np.array([undistorted_center[1], 1])

    box_x = box_x / np.linalg.norm(box_x)
    box_y = box_y / np.linalg.norm(box_y)

    # These are the proper (it seems) angles, with signs.
    # Directions align with typical cv (+x right, +y down)
    print("Signed angles:",
          np.rad2deg(np.arctan2(box_x[0], box_x[1])),
          np.rad2deg(np.arctan2(box_y[0], box_y[1])))

    cv2.imshow("Original", frame)
    #  cv2.imshow("Masked", masked)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
