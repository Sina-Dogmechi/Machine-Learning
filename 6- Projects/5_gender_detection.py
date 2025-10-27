# delete tensorflow warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
from mtcnn import MTCNN
from joblib import load
import glob
import numpy as np


model = load("gender_detector.z")

detector = MTCNN()

def face_detector(img):
    try:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = detector.detect_faces(rgb_img)[0]
        x, y, w, h = out['box']

        return img[y:y+h, x:x+w], x, y, w, h
    
    except:
        pass

for item in glob.glob("test_images_2\\*"):
    
    img = cv2.imread(item)
    face, x, y, w, h = face_detector(img)

    if face is None:
        continue

    else:
        face = cv2.resize(face, (32, 32))
        face = face / 255
        face = face.flatten()
        face = np.array([face])

        out = model.predict(face)[0]
        
        if out == 'male':
            cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, out, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        elif out == 'female':
            cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(img, out, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
        cv2.imshow("Image", img)

        if cv2.waitKey(0) == ord('q'):
            break

cv2.destroyAllWindows()