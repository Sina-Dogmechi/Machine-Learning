import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_percentage_error


def load_house_attributes(datapath):
    cols = ['bedrooms', 'bathrooms', 'area', 'zipcode', 'price']

    df = pd.read_csv(datapath, sep=' ', header=None, names=cols)

    zipcodes = df['zipcode'].value_counts().keys().to_list()
    counts = df['zipcode'].value_counts().to_list()
    
    # zipcode is a nominal variable 
    # we want to ommit zipcodes with counts less than 25
    for zipcode, count in zip(zipcodes, counts):
        if count < 25:
            indx = df[df['zipcode'] == zipcode].index
            # print(indx)
            df.drop(indx, inplace=True)
    
    # print(df.shape)
    return df       

def preprocess_house_attribute(train, test):
    continuous = ['bedrooms', 'bathrooms', 'area']

    sc = StandardScaler()

    traincontinuous = sc.fit_transform(train[continuous])
    testcontinuous = sc.transform(test[continuous])

    # print(traincontinuous.shape)
    # print(testcontinuous.shape)
    """
    encoder = OneHotEncoder(sparse_output=False)

    traincategorical = encoder.fit_transform(np.array(train['zipcode']).reshape(-1, 1))
    testcategorical = encoder.fit_transform(np.array(test['zipcode']).reshape(-1, 1))
    """
    encoder = LabelBinarizer()

    traincategorical = encoder.fit_transform(train['zipcode'])
    testcategorical = encoder.fit_transform(test['zipcode'])
    
    trainX = np.hstack([traincontinuous, traincategorical])
    testX = np.hstack([testcontinuous, testcategorical])

    # print(trainX.shape)
    # print(testX.shape)
    return trainX, testX

df = load_house_attributes(r'dataset\HousesInfo.txt')

train, test = train_test_split(df, test_size=0.2, random_state=42)

trainX, testX = preprocess_house_attribute(train, test)


# Label Normalize
max_price = train['price'].max()
trainY = train['price'] / max_price
testY = test['price'] / max_price

# model = LinearRegression()
model = SGDRegressor(tol=0.000001)
model.fit(trainX, trainY)

# print(model.coef_)


# mean absolute percentage error (MAPE)
"""
preds = model.predict(testX)
diff = preds - testY
percentDiff = (diff/testY) * 100
absPercentDiff = np.abs(percentDiff)

mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

print('mean: {:.2f}, std: {:.2f}'.format(mean, std))
"""

preds = model.predict(testX)

mape = mean_absolute_percentage_error(testY, preds)

print("MAPE: {:.2f}".format(mape*100))