# delete tensorflow warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
from mtcnn import MTCNN


detector = MTCNN()

img = cv2.imread("maryam.webp")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

out = detector.detect_faces(rgb_img)[0]
x, y, width, height = out['box']

confidence = out['confidence']
text = "Prob: {:.2f}".format(confidence*100)

cv2.rectangle(img, (x,y), (x+width, y+height), (0, 255, 0), 2)

cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

KP = out['keypoints']
for key, value in KP.items():
    cv2.circle(img, value, 8, (0, 0, 255), -1)

cv2.imshow('Image', img)

cv2.waitKey(0)

cv2.destroyAllWindows()