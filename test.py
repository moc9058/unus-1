import cv2
import numpy as np

video = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while True:
    ret, image = video.read()
    cv2.imshow("Test image",image)
    print(image.shape)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



video.release()
cv2.destroyAllWindows()
