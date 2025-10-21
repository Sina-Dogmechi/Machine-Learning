import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import zero_one_loss


def load_data():
    dataset = pd.read_csv(r"Week2-Datasets\iris.data", header=None,
                        names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'])

    # nrows, ncols = dataset.shape

    # print("number of samples: {}, number of features: {}".format(nrows, ncols))

    data = dataset.iloc[:, :4]
    label = dataset.iloc[:, 4]

    # x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)

    return x_train, x_test, y_train, y_test


def training():
    clf = KNeighborsClassifier(5)

    clf.fit(x_train, y_train)

    return clf


def result():
    y_pred = clf.predict(x_test)

    acc = accuracy_score(y_test, y_pred)

    print("accuracy: {:.2f}".format(acc*100))

    loss = zero_one_loss(y_test, y_pred)

    print("loss: {:.2f}".format(loss*100))



x_train, x_test, y_train, y_test = load_data()

clf = training()

result()