# -*- coding: utf-8 -*-
"""
Created on Thu May 28 18:09:45 2020

@author: Nooreldean Koteb
"""

#Recurrent Neural Network
#Part 1 - Data Preproccessing
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

#Feature scaling - This time were gonna use normalization
from sklearn.preprocessing import MinMaxScaler
sc =  MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

#Creating a data structure with 60 timesteps and 1 output
x_train = []
y_train = []
for i in range(60, 1258):
    x_train.append(training_set_scaled[i-60:i, 0])
    #y_train.append(training_set_scaled[[i, 0]])
    y_train.append(training_set_scaled[[i][0]])
    
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshaping data - can add more values if wanted to
#Change 1 to the number of values being used to predict
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


#Part 2 - Building the RNN
#Importing keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

#Initializing the RNN
regressor = Sequential()

#Adding the first LSTM layer and some dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(.2))

#Adding the second LSTM layer and some dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(.2))

#Adding the third LSTM layer and some dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(.2))

#Adding the fourth LSTM layer and some dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(.2))

#Adding the output layer
regressor.add(Dense(units = 1))

#Compiling the RNN
#rmsProp optimizer tends to be recommended for RNNs
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Fitting the RNN to the training set
regressor.fit(x_train, y_train, epochs = 100, batch_size = 32)#, max_queue_size = 100, workers = 12)


#Part 3 - Making the predictions and visualising the results
#Getting the real stock price of 2017
#Importing test set
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

#Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[((len(dataset_total) - len(dataset_test))- 60):].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
#Creating a data structure with 60 timesteps and 1 output
x_test = []
for i in range(60, 80):
    x_test.append(inputs[i-60:i, 0])
    
x_test= np.array(x_test)

#Reshaping data - can add more values if wanted to
#Change 1 to the number of values being used to predict
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Prediction
predicted_stock_price = regressor.predict(x_test)
#Unscale the values
predicted_stock_price = sc.inverse_transform(predicted_stock_price) 


#Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price Jan 2017')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price Jan 2017')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


# #Evaluating
# #Calculating RMSE (Not very relevant here since we care about the direction of the price, not the exact price)
# import math
# from sklearn.metrics import mean_squared_error
# rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
# #Consider dividing RMSE by range of google stock price (around 800) to get relative error


# Improving
# 1. More training data
# 2. Increasing the timesteps
# 3. Adding some other indicators
# 4. Adding more LSTM layers
# 5. Adding more Neurons in the LSTM layers


#Tuning
#Same as ANN and CNN except replace scoring = 'accuracy'
#to scoring = 'neg_mean_squared_error'








