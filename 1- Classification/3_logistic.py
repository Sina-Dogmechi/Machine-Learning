# Logistic Regression (classification)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


dataset = pd.read_csv('dataset/diabetes.csv')

zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for coloumn in zero_not_accepted:
    dataset[coloumn] = dataset[coloumn].replace(0, np.nan)
    mean = int(dataset[coloumn].mean(skipna=True))
    dataset[coloumn] = dataset[coloumn].replace(np.nan, mean)

data = dataset.iloc[:, :8]
label = dataset.iloc[:, 8]

x_train, x_test, y_train, y_test =train_test_split(data, label, test_size=0.2, random_state=42)

sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

clf = LogisticRegression()

sgd_clf = SGDClassifier(loss='log_loss')

# clf.fit(x_train, y_train)
sgd_clf.fit(x_train, y_train)

# y_pred = clf.predict(x_test)
y_pred = sgd_clf.predict(x_test)

# acc = accuracy_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print('accuracy: {:.2f}'.format(acc*100))

"""
acc = clf.score(x_test, y_test)
print('accuracy in score:', acc)
"""

prob = sgd_clf.predict_proba(x_test)
print(prob)