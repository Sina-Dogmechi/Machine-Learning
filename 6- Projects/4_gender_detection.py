# delete tensorflow warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
from mtcnn import MTCNN
from joblib import dump
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


detector = MTCNN()

def face_detector(img):
    try:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = detector.detect_faces(rgb_img)[0]
        x, y, w, h = out['box']

        return img[y:y+h, x:x+w]
    
    except:
        pass


data = []
labels = []

for i, item in enumerate(glob.glob("dataset\\Gender\\*\\*")):
    
    img = cv2.imread(item)
    face = face_detector(img)

    if face is None:
        continue

    else:
        face = cv2.resize(face, (32, 32))
        face = face / 255
        face = face.flatten()
        data.append(face)

        label = item.split("\\")[-2]
        labels.append(label)
    
    if i % 100 == 0:
        print("[info]: {}/3732 processed".format(i))
    

data = np.array(data)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# loss is SVM (we can say best classifier)
clf = SGDClassifier()

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = accuracy_score(y_test, y_pred)

print("accuracy: {:.2f}".format(acc*100))

# dump(clf, "gender_detector.z")