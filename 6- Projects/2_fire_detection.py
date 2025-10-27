import cv2
import numpy as np
import glob
from joblib import load

model = load("fire_detector.z")

for item in glob.glob(r"test_images\*"):

    img = cv2.imread(item)
    re_img = cv2.resize(img, (32, 32))
    re_img = re_img / 255
    re_img = re_img.flatten()

    # Because basic ML algorithms work with 2-D
    re_img = np.array([re_img])
    # print(re_img.shape)

    label = model.predict(re_img)[0]
    
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

    cv2.imshow("Image", img)

    cv2.waitKey(0)

cv2.destroyAllWindows()

