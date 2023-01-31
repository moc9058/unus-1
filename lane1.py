# -*- coding: utf-8 -*-
import cv2
import numpy as np
import Vision.cv_util_func1 as cv_util
import Vision.cam_util_func as cam_util
import sys


# # image
# lane_detection = cv_util.libLANE()
# image = cv2.imread('./flower_images/flower_1.png')
# result = lane_detection.lane(image)
# cv2.imshow('result', result)
# cv2.waitKey(0)

def image():
    # image
    lane_detection = cv_util.libLANE()
    image = cv2.imread('./flower_images/flower_4.jpg')
    result = lane_detection.lane(image)

    cv2.imshow('result', result)
    cv2.waitKey(0)


# video
def video():

    # cap = cv2.VideoCapture('./record/4.mp4')
    # cap = cv2.VideoCapture('./record/230127/4_l.mp4')
    # cap = cv2.VideoCapture('./record/230127/7_r.mp4')
    cap = cv2.VideoCapture('./test_videos/230120/1.mp4')
    # lane_detection = cv_util.libLANE()

    while (cap.isOpened()):
        ret, image = cap.read()
        # height, width, channel=image.shape
        # image_ul = image[:height//2, :width//2, :]
        # image_ur = image[:height//2, width//2:, :]
        # image_ll = image[height//2:, :width//2, :]
        # image_lr = image[heiqght//2:, width//2:, :]
        # cv2.imshow('Image', image_ul)

        # detected = lane_detection.lane(image)
        
        # cv2.imshow('image', image)
        # cv2.imshow('result', detected)
        # cv2.imshow('hls', cv2.cvtColor(image, cv2.COLOR_BGR2HLS_FULL))
        cv2.imshow('hsv', cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL))
        key = cv2.waitKey(5)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('w'):
            cv2.waitKey(0)

    # Release
    cap.release()
    cv2.destroyAllWindows()
    sys.exit()

image()

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