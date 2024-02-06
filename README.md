# Understanding and Implementing Linear Regression for Stock Price Prediction

Predicting the closing stock price for a given stock, here exemplified by Apple Inc. (AAPL), using historical data spanning the last 10 years. The code utilizes the Linear Regression algorithm to establish a relationship between various features derived from stock data and the adjusted closing prices.

## Importing Libraries:
The script starts by importing necessary libraries such as pandas for data manipulation, numpy for numerical operations, matplotlib for plotting graphs, datetime for handling dates, yfinance for fetching stock data, and scikit-learn for machine learning functionalities.

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
```

## Setting Parameters and Fetching Data:
The script sets parameters like the stock symbol ('AAPL') and the date range for historical data (10 years). It then downloads the historical stock data using the yfinance library.
```
stock = 'AAPL'
start_date = dt.date.today() - dt.timedelta(days=3650)  # 10 years of data
end_date = dt.date.today()
data = yf.download(stock, start_date, end_date)
```
## Data Preprocessing:
Irrelevant columns, such as 'Adj Close', are dropped from the dataset. The features (X) are defined as all columns except 'Close', and the target variable (y) is set as 'Adj Close'.
```
data = data.drop(columns=['Adj Close'])
X = data.drop(['Adj Close'], axis=1)
y = data['Adj Close']
```
## Splitting Data into Training and Testing Sets:
The dataset is divided into training and testing sets using the train_test_split function from scikit-learn.
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
```
## Training the Linear Regression Model:
A linear regression model is instantiated and trained using the training data.
```
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)
```
## Model Evaluation:
The script evaluates the model's performance by calculating the R-squared score on the testing set.
```
score = regression_model.score(X_test, y_test)
print(f"The score for our model is: {score}")
```
## Predicting the Next Day's Closing Price:
The script predicts the next day's closing price using the latest available data.
```
latest_data = data.tail(1).drop(['Adj Close'], axis=1)
next_day_price = regression_model.predict(latest_data)[0]
print(f"The predicted price for the next trading day is: {next_day_price}")
```
It emphasizes the importance of data preprocessing, splitting data for training and testing, model training, and evaluation. This code serves as an illustrative example for data science students, showcasing the practical implementation of linear regression in a financial context. 
