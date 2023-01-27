import cv2
import numpy as np

video = cv2.VideoCapture(0)
ret, image = video.read()
print(ret)
# cv2.imshow("Test image",image)
# print(image.shape)
# cv2.waitKey(0)
# video.release()
# cv2.destroyAllWindows()
