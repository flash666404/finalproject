import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import Sequential
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['unix_timestamp'] = df['Date'].astype('int64') // 10**9
    df.unix_timestamp = pd.to_datetime(df.unix_timestamp, unit='s')
    df.index = df.unix_timestamp
    df = df.resample('D').mean()
    df = df.dropna()
    return df

def preprocess_data(data, window_size=30):
    price = data['Weighted_Price']
    X = price
    size = int(len(X) * 0.7)
    train_df, test_df = X[0:size], X[size:len(X)]
    training_values = train_df.values
    training_values = np.reshape(training_values, (len(training_values), 1))

    # Stationary transformation
    stationary_values = np.zeros_like(training_values)
    for i in range(len(training_values) - window_size):
        stationary_values[i + window_size] = training_values[i + window_size] - training_values[i]
    x_train = stationary_values[window_size - 1 : -1]
    y_train = training_values[window_size:]

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_train = np.reshape(x_train, (len(x_train), 1, 1))
    return x_train, y_train, test_df, scaler

def build_model():
    model = Sequential()
    model.add(LSTM(10, input_shape=(None, 1), activation="relu", return_sequences=True))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model

def train_model(model, x_train, y_train, epochs=50, batch_size=32):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

def predict_price(model, test_data, scaler):
    test_values = test_data.values
    test_values = np.reshape(test_values, (len(test_values), 1))
    test_values = scaler.transform(test_values)
    test_values = np.reshape(test_values, (len(test_values), 1, 1))
    predicted_price = model.predict(test_values)
    predicted_price = np.reshape(predicted_price, (len(predicted_price), 1))
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price

def plot_predictions(test_df, predicted_price):
    test_df = test_df.to_frame()
    test_df = test_df.reset_index()
    df1 = pd.read_csv('BTC-USD.csv')
    test_date = df1['Date'].tail(549)
    test_date = pd.DataFrame(test_date, columns=['Date'])
    test_date = test_date.reset_index()
    test_date.drop("index", axis=1, inplace=True)
    predicted_df = pd.DataFrame(predicted_price, columns=['predictions'])
    test_df = pd.concat([test_df['Weighted_Price'], predicted_df['predictions'], test_date['Date']], axis=1)
    test_df.set_index('Date', inplace=True)

    plt.figure(figsize=(20, 13))
    ax = plt.gca()
    test_df.Weighted_Price.plot(color='blue', label='Real Price')
    test_df.predictions.plot(color='r', ls='--', label='Predicted Weighted_Price')
    plt.legend()
    plt.ylabel('Price')
    plt.title('BTC Price Prediction')
    plt.show()

def evaluate_model(test_df, predicted_price):
    mae = np.mean(np.abs(test_df.Weighted_Price - predicted_price))
    mape = np.mean(np.abs((test_df.Weighted_Price - predicted_price) / test_df.Weighted_Price)) * 100
    rmse = np.sqrt(np.mean((test_df.Weighted_Price - predicted_price) ** 2))
    r_squared = r2_score(test_df.Weighted_Price, predicted_price)

    metrics_df = pd.DataFrame({
        'MAE': [mae],
        'MAPE': [mape],
        'RMSE': [rmse],
        'R^2': [r_squared]
    })

    print(metrics_df)

def main():
    file_path = 'BTC-USD.csv'
    df = load_data(file_path)
    x_train, y_train, test_df, scaler = preprocess_data(df)
    model = build_model()
    train_model(model, x_train, y_train, epochs=50, batch_size=32)
    predicted_price = predict_price(model, test_df, scaler)
    plot_predictions(test_df, predicted_price)
    evaluate_model(test_df, predicted_price)

if __name__ == "__main__":
    main()
