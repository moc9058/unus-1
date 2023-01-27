# -*- coding: utf-8 -*-
import cv2
import numpy as np
import Vision.cv_util_func1 as cv_util
import Vision.cam_util_func as cam_util
import sys

'''
# image
lane_detection = cv_util.libLANE()
image = cv2.imread('./test_images/24_19-48-02.png')
result = lane_detection.lane(image)

cv2.imshow('result', result)
cv2.waitKey(0)

'''

# video
cv2.namedWindow('result')
cv2.setMouseCallback('result', mouse_pos_BGR)
cap = cv2.VideoCapture('./test_videos/230120/1.mp4')
lane_detection = cv_util.libLANE()

while (cap.isOpened()):
    ret, image = cap.read()
    height, width, channel=image.shape
    image_ul = image[:height//2, :width//2, :]
    image_ur = image[:height//2, width//2:, :]
    image_ll = image[height//2:, :width//2, :]
    image_lr = image[height//2:, width//2:, :]
    # cv2.imshow('Image', image_ul)

    detected = lane_detection.lane(image)
    cv2.imshow('result', detected)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release
cap.release()
cv2.destroyAllWindows()
sys.exit()

'''
# cam
cam = cam_util.libCAMERA()
ch0, ch1 = cam.initial_setting(cam0port=1, cam1port=2, capnum=2)
lane_detection = cv_util.libLANE()
q
while True:
    _, frame0, _, frame1 = cam.camera_read(ch0, ch1)
    cam.image_show(frame0, frame1)

    result = lane_detection.lane(frame1)

    cv2.imshow('result', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # if cam.loop_break():
    #     break

# Release
cv2.destroyAllWindows()
'''