#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Reading the dataset
dataset = pd.read_csv("data.csv")

dataset.head()

x = dataset[['n','T']]
y = dataset['K']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)

from sklearn.linear_model import LinearRegression

#Fitting the Multiple Linear Regression model
mlr = LinearRegression()  
mlr.fit(x_train, y_train)

#Intercept and Coefficient
print("Intercept: ", mlr.intercept_)
print("Coefficients:")
list(zip(x, mlr.coef_))

#Prediction of test set
y_pred_mlr= mlr.predict(x_test)
#Predicted values
print("Prediction for test set: {}".format(y_pred_mlr))



print("The value for reaction rate for temperature 923 and 8 carbon atoms is: ", mlr.predict([[8,923]]))
print("\n The value for reaction rate for temperature 973 and 8 carbon atoms is: ", mlr.predict([[8,973]]))
print("\n The value for reaction rate for temperature 1023 and 8 carbon atoms is: ", mlr.predict([[8,1023]]))
print("\n The value for reaction rate for temperature 1073 and 8 carbon atoms is: ", mlr.predict([[8,1073]]))