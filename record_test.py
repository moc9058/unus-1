import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"]="0"

import cv2
import datetime
import sys

num_camera = 1
if num_camera == 2:
    # left
    video1 = cv2.VideoCapture(0,cv2.CAP_MSMF)
    video1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    video1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # right
    video2 = cv2.VideoCapture(1,cv2.CAP_MSMF)
    video2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    video2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    num_files = len(os.listdir("./record/230127"))
    out1 = cv2.VideoWriter(f"./record/230127/{num_files//2+1}_l.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20, (1920, 1080))
    out2 = cv2.VideoWriter(f"./record/230127/{num_files//2+1}_r.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20, (1920, 1080))
    while True:
        ret1, image1 = video1.read()
        ret2, image2 = video2.read()
        image = image2
        cv2.imshow("Webcam1", image1)
        cv2.imshow("Webcam2", image2)
        out1.write(image1)
        out2.write(image2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video1.release()
    video2.release()
    out1.release()
    out2.release()
else:
    video = cv2.VideoCapture(0, cv2.CAP_MSMF)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    num_files = len(os.listdir("./record/230127"))
    out = cv2.VideoWriter(f"./record/{num_files}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20, (1920, 1080))
    while True:
        ret, image = video.read()
        cv2.imshow("Webcam1", image)
        out.write(image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    out.release()
cv2.destroyAllWindows()
sys.exit()