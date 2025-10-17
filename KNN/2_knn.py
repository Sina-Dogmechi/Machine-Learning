import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def load_data():

    dataset = pd.read_csv("Week2-Datasets/diabetes.csv")

    ## preprocessing
    # Missing Values: replace zero values with mean
    zero_not_accepted = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    for coloumn in zero_not_accepted:
        dataset[coloumn] = dataset[coloumn].replace(0, np.nan)
        mean = int(dataset[coloumn].mean(skipna=True))
        dataset[coloumn] = dataset[coloumn].replace(np.nan, mean)


    data = dataset.iloc[:, :8]
    label = dataset.iloc[:, 8]

    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)

    # Normalizing
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.transform(x_test)

    return x_train, x_test, y_train, y_test


def algorithm():

    clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean")

    clf.fit(x_train, y_train)

    return clf


def show_results():

    y_pred = clf.predict(x_test)

    acc = accuracy_score(y_test, y_pred)

    print("accuracy: {:.2f}".format(acc*100))


x_train, x_test, y_train, y_test = load_data()

clf = algorithm()

show_results()