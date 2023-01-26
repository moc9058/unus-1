import cv2
import Vision.cam_util_func as cam_util

cam = cam_util.libCAMERA()
ch0, ch1 = cam.initial_setting(cam0port=0, cam1port=1, capnum=2)

while True:
    _, frame0, _, frame1 = cam.camera_read(ch0, ch1)
    cam.image_show(frame0, frame1)

    if cam.loop_break():
         break

# Release
cv2.destroyAllWindows()