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

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()

im = cv2.imread(args.filename)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY |
        cv2.THRESH_OTSU)

hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
#  hsv_blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
hsv_blurred = hsv
ret, thresh_h = cv2.threshold(hsv_blurred[:, :, 0], 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)
ret, thresh_s = cv2.threshold(hsv_blurred[:, :, 1], 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)
ret, thresh_v = cv2.threshold(hsv_blurred[:, :, 2], 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# threshold on each of the channels, see what happens

img = gray.copy()
cimg = im.copy()
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)
circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
cv2.imshow('detected circles',cimg)




cv2.imshow("Image", im)
cv2.imshow("thresh bw", thresh)
cv2.imshow("thresh hue", thresh_h)
cv2.imshow("thresh sat", thresh_s)
cv2.imshow("thresh val", thresh_v)

cv2.waitKey(0)
cv2.destroyAllWindows()
