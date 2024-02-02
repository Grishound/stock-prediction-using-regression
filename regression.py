import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

stock = 'AAPL'
start_date = dt.date.today() - dt.timedelta(days=3650)  # 10 years of data
end_date = dt.date.today()
data = yf.download(stock, start_date, end_date)
data = data.drop(columns=['Adj Close'])

X = data.drop(['Close'], axis=1)
y = data['Adj Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

regression_model = LinearRegression()
regression_model.fit(X_train, y_train)
intercept = regression_model.intercept_
print(f"The intercept for our model is: {intercept}")

score = regression_model.score(X_test, y_test)
print(f"The score for our model is: {score}")
latest_data = data.tail(1).drop(['Close'], axis=1)
next_day_price = regression_model.predict(latest_data)[0]
print(f"The predicted price for the next trading day is: {next_day_price}")
