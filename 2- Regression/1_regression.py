# Simple Linear Regression

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


dataset = pd.read_csv("Week2-Datasets/Dataset.csv")

# print(dataset.shape)
# print(dataset.info())

data = dataset.iloc[:, :-1]
label = dataset.iloc[:, 1]

"""
plt.scatter(data, label)
plt.title("Hours vs. Scores")
plt.xlabel('hours')
plt.ylabel("scores")
plt.show()
"""

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)

regressor = LinearRegression()

regressor.fit(x_train, y_train)

# print('m: ', regressor.coef_)
# print('b: ', regressor.intercept_)
print("y = {:.2f}x + {:.2f}".format(regressor.coef_[0], regressor.intercept_))

y_pred = regressor.predict(x_test)

# df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# print(df)

print('MAE:', mean_absolute_error(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))


plt.scatter(x_test, y_test, label='actual')
plt.title("Hours vs. Scores")
plt.xlabel('hours')
plt.ylabel("scores")

plt.plot(x_test, y_pred, 'r', label='Predicted')

plt.legend()

plt.show()