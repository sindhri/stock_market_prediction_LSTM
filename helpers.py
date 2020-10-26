import numpy as np 
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM

# scale, then train-test split
def scale_train_test_split_preprocess(original, training_ratio, step_size):
    
    # scale the data
    data = np.reshape(original.values, (len(original),1)) # 1664
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    
    train = int(len(data) * training_ratio)
    test = len(data) - train
    train, test = data[0:train,:], data[train:len(data),:]
    # preprocess to create X and Y based on step_size
    trainX, trainY = build_XY(train, step_size)
    testX, testY = build_XY(test, step_size)
    # reshape the data for LSTM
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    return trainX, trainY, testX, testY, scaler

# use step_size to create X and Y for time series analysis
def build_XY(dataset, step_size):
    data_X, data_Y = [], []
    for i in range(len(dataset)-step_size-1):
        a = dataset[i:(i+step_size), 0]
        data_X.append(a)
        data_Y.append(dataset[i + step_size, 0])
    return np.array(data_X), np.array(data_Y)

# LSTM MODEL
def build_LSTM(step_size):
    model = Sequential()
    model.add(LSTM(32, input_shape=(1, step_size), return_sequences = True))
    model.add(LSTM(16))
    model.add(Dense(1))
    model.add(Activation('linear'))
    return model

# Make prediction and calculate training and test results
def make_prediction(model, scaler, trainX, trainY, testX, testY):
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # inverse the scale to the actual values
    trainPredict_reversed = scaler.inverse_transform(trainPredict)
    trainY_reversed = scaler.inverse_transform([trainY])
    testPredict_reversed = scaler.inverse_transform(testPredict)
    testY_reversed = scaler.inverse_transform([testY])

    # TRAINING RMSE    
    trainScore = math.sqrt(mean_squared_error(trainY_reversed, np.transpose(trainPredict_reversed)))
    print('Train RMSE: %.2f' % (trainScore))
    # TEST RMSE
    testScore = math.sqrt(mean_squared_error(testY_reversed, np.transpose(testPredict_reversed)))
    print('Test RMSE: %.2f' % (testScore))
    return trainPredict_reversed, testPredict_reversed
    
# make the plot data
def result_plot(series_date, original, step_size, trainPredicted, testPredicted):
    trainPredictedPlot = np.empty_like(original)
    trainPredictedPlot[:] = np.nan
    trainPredictedPlot[step_size:len(trainPredicted)+step_size] = trainPredicted[:,0]
    testPredictedPlot = np.empty_like(original)
    testPredictedPlot[:] = np.nan
    testPredictedPlot[len(trainPredicted)+(step_size*2)+1:len(original)-1] = testPredicted[:,0]
    # plot the predicted and original results
    plt.figure(figsize=(20,10))
    plt.plot(series_date, original, 'g', label = 'original dataset')
    plt.plot(series_date, trainPredictedPlot, 'r', label = 'training set')
    plt.plot(series_date, testPredictedPlot, 'b', label = 'predicted stock price/test set')
    plt.legend(loc = 'upper left')
    plt.xlabel('Time')
    plt.ylabel('Stock Value')
    plt.show()
    