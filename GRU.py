import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras 

dataset = pd.read_csv('train.csv')
X = dataset.iloc[:,1:2].values

# Always normalize the data as these process are hihgly computive 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)


# Breaking down the data to convert into to feedable data to the RNN, We form sequence of 60
# values with the 61st value being the output data
x_train = []
y_train = []
for i in range(60,1259):
    x_train.append(X[i-60:i,0])
    y_train.append(X[i,0])
    
x_train = np.array(x_train)
y_train = np.array(y_train)

# Reshaping the input data as it has to 3 Dimensional
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


    
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

# Creating the GRU model 
classifier = Sequential()

# A layer with 100 neurons with 60 inputs
classifier.add(GRU(100,input_shape=(60,1),return_sequences=True, activation='tanh'))
#Dropout to reduce overfitting of the model, kills random neurons but this increses the time
# taken by the neural network to converge at a single point
classifier.add(Dropout(0.2))

# 2nd layer, here need not mention the input as it is automatically assigned
classifier.add(GRU(100,return_sequences=True, activation = 'tanh'))
classifier.add(Dropout(0.2))

# 3rd layer, the last LSTM layer so not including 'True' for the return_sequences as I just need
# a 2D matrix to input it to the following ANN model
classifier.add(GRU(100,return_sequences=False, activation = 'tanh'))
classifier.add(Dropout(0.2))

# Output node with just one neuron that predicts the stock price (no activation units as we need
# the value directly)
classifier.add(Dense(1))

# gives the summary of the neural network
classifier.summary()

# Determining the optimizers for the built neural network
# metrics is not required, not considered
classifier.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])

#Fitting the model(Neural network) to our dataset
classifier.fit(x_train, y_train, batch_size = 30, epochs=100)



# Inputting the test data whose results are to be predicted 
test_set = pd.read_csv('testset.csv')
Y= test_set.iloc[:,1:2].values

Y = sc_X.transform(Y)

x_test = []
y_test = []
for i in range(60,125):
    x_test.append(Y[i-60:i,0])
    y_test.append(Y[i,0])
    
x_test = np.array(x_test)
y_test = np.array(y_test)


x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

y_pred = classifier.predict(x_test)

# Inverse tranform to negate the normalization operation performed earlier
y_test = sc_X.inverse_transform(y_test)
y_pred = sc_X.inverse_transform(y_pred)

# To plot the actual and the predicted data to compare with each other
import matplotlib.pyplot as plt

plt.plot(y_test,color = 'red', label = 'Real Price')
plt.plot(y_pred, color = 'blue', label = 'Predicted Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()









