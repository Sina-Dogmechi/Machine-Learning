import cv2
from sklearn.model_selection import train_test_split
import glob
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data():
    
    data_list = []
    labels = []

    for i, address in enumerate(glob.glob("dataset\\fire_dataset\\*\\*")):

        img = cv2.imread(address)
        img = cv2.resize(img, (32, 32))
        img = img / 255.0
        img = img.flatten()

        data_list.append(img)

        label = address.split("\\")[-1].split(".")[0]
        labels.append(label)

        if i % 100 ==0:
            print("{}/{} processed".format(i, 1000))
        

    data_list = np.array(data_list)
    
    x_train, x_test, y_train, y_test = train_test_split(data_list, labels, test_size=0.2)

    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = load_data()

clf = KNeighborsClassifier()

clf.fit(x_train, y_train)

# acc = clf.score(x_test, y_test)

# print("accuracy: {:.2f}".format(acc*100)) # after execute: acc = 88.0

# dump(clf, "fire_detector.z")


y_pred = clf.predict(x_test)

acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="fire")
recall = recall_score(y_test, y_pred, pos_label="fire")
f_score = f1_score(y_test, y_pred, pos_label="fire")

print("accuracy: {:.2f}".format(acc*100))
print("precision: {:.2f}".format(precision*100))
print("recall: {:.2f}".format(recall*100))
print("f_score: {:.2f}".format(f_score*100))