import numpy as np
import cv2
import sys

img1 = cv2.imread('picture3.png')
img2 = cv2.imread('picture4.png')
image = [img1, img2]
cv2.imshow("Picture 1", img1)
cv2.imshow("Picture 2", img2)

stitcher = cv2.Stitcher.create()
status, pano = stitcher.stitch(image)

if status != cv2.Stitcher_OK:
    print("Stitch failed!")
    sys.exit()
else:
    cv2.imshow("Stitched picture", pano)

cv2.waitKey(0)
cv2.destroyAllWindows()