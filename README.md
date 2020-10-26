# Stock market prediction of Apple Inc
Used the stock market price of Apple Inc 2011-2016 to predict 2016-2017 price with an average of $2.80 difference between prediction and actual value
Inspired by https://github.com/NourozR/Stock-Price-Prediction-LSTM

## 1. EDA
* Used stock from Apple Inc 2011-2017
* Built a LSTM model to predict the future price
* Three potential indicators: average of open/high/low/close, average of high/low/close, just close. They produce very similar results
<img src = "https://github.com/sindhri/animal_recognition/blob/main/img/img1.png" width = "800">

## 2. LSTM
* Split the test by 75/25 training vs. test. 
* Apply MinMax scaler on the data
* step size = 1
* 200 epoch
* Only took a few minutes to train
<img src = "https://github.com/sindhri/animal_recognition/blob/main/img/model.png" width = "250">
<img src = "https://github.com/sindhri/animal_recognition/blob/main/img/img3.png" width = "500">


## 3. Prediction
The RMSE (root mean square error) of the test data is 2.80, which means the predicte value and the actual value had an average of $2.80 difference. 
<img src = "https://github.com/sindhri/animal_recognition/blob/main/img/img2.png" width = "800">


## 4. Future directions
* test more recent data
* test different step size
* include validation data
* Make an API