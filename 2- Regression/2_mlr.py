# Multiple Linear Regression (MLR)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv('dataset/petrol_consumption.csv')

# print(dataset.shape)
# print(dataset.info())

# plt.scatter(dataset['Population_Driver_licence(%)'], dataset['Petrol_Consumption'])
# plt.show()

data = dataset[['Petrol_tax', 'Average_income', 'Paved_Highways', 'Population_Driver_licence(%)']]
label = dataset['Petrol_Consumption']

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)

regressor = LinearRegression()

regressor.fit(x_train, y_train)

# print(regressor.coef_)
# print(regressor.intercept_)

coef_df = pd.DataFrame(regressor.coef_, index=data.columns ,columns=['Coefficients'])
# print(coef_df)

y_pred = regressor.predict(x_test)

df = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
# print(df)

print('MAE:', mean_absolute_error(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))