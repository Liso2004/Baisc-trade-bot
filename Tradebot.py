from typing import final
import numpy as np #setting up the project 
import pandas as pd 
import matplotlib.pyplot as plt 
from keras.models import sequential 
from keras.layers import Dense , LSTM 
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv("all _stock_5yr...csv")#data sheet to be used to train the model 
data.head()
all_stock_names =data['name'].unique()
print(all_stock_names) 

# 1 getting the stock name 
stock_name= input("Enter a Stock Price Name: ")
# 2 extracting all the data having the same name as stock enterned 
all_data = data['name']== stock_name
# 3 putting all the rows of specific stock in a variable 
final_data = data [all_data ]
# 4 printing the first 5 rows of the stock data of specfic stock name 
final_data.head()
#plotting the data vs the close market stock price 
final_data.plot ('data','close',color="red")
# Extract only top 60 rows to make a little more clearer to view 
New_data = final_data.head(60)
#Plotting data vs the close market stock price 
New_data.plot ('data','close',color="green")

plt.show()

#filter out the closing market price data i
close_data = final_data.filter(['close'])

#convert the data into array for easy evaluation 
dataset= close_data.values

#scale/normalize the data to make all values between 0 to 1 
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

#creating training data size : 70% of the data 
Training_data_len = math.ceil(len(dataset)*.7)
train_data = scaled_data[0:Training_data_len , :]

#separating the data into X and Y data 
x_train_data=[]
y_train_data=[]
for i in range (60,len(train_data)):
    x_train_data =list(x_train_data)
    y_train_data = list(y_train_data)
    x_train_data.append(train_data[i-60:i,0])
    y_train_data.append(train_data[i,0])

    #converting the training x and y values to numpy arrays 

#https://www.askpython.com/python/examples/stock-price-prediction-python