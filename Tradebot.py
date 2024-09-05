#https://www.askpython.com/python/examples/stock-price-prediction-python


from typing import final
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, LSTM 
import math
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
    x_train_data1,y_train_data1 = np.array(x_train_data) , np.array(y_train_data)

    #Reshaping training s nad y data to make tyhe calculations easier 
    x_train_data2 = np.reshape(x_train_data1 , (x_train_data1.shape[0] , x_train_data1.shape[1],1))

    #Building LSTM Model 
    model = Sequential()
    model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data2.shape[1],1)))
    model.add(LSTM(units=50,return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    #Compiling the model
    model.compile(optimizer='adam',loss='mean_squared_error')
    model.fit(x_train_data2 , y_train_data1 , batch_size =1 , epochs =1)
    
# 1. Creating a dataset for testing
test_data = scaled_data[Training_data_len - 60: , : ]
x_test = []
y_test =  dataset[Training_data_len : , : ]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
 
# 2.  Convert the values into arrays for easier computation
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
 
# 3. Making predictions on the testing data
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
print(rmse)

train = data[:Training_data_len]
valid = data[Training_data_len:]
 
valid['Predictions'] = predictions
 
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close')
 
plt.plot(train['close'])
plt.plot(valid[['close', 'Predictions']])
 
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
 
plt.show()