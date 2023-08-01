import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score

def load_data(file_path):
    df = pd.read_csv(file_path)
    window_size = 30
    df['Weighted_Price'] = df['Weighted_Price'].rolling(window=window_size).mean()
    df.dropna(inplace=True)
    return df

def make_data_stationary(data, window_size=30):
    price_diff = data['Weighted_Price'].diff().dropna()
    price_diff_rolling_mean = price_diff.rolling(window=window_size).mean()
    data['Weighted_Price'] = price_diff - price_diff_rolling_mean
    data.dropna(inplace=True)
    return data

def train_test_split_data(data, test_size=0.3):
    price = data['Weighted_Price']
    X = price.values
    size = int(len(X) * (1 - test_size))
    train, test = X[:size], X[size:]
    return train, test

def fit_arima_model(train_data, order=(5, 1, 0)):
    history = [x for x in train_data]
    predictions = []
    for t in range(len(test_data)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        output = model_fit.forecast()
        result = output[0]
        predictions.append(result)
        obs = test_data[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (result, obs))
    return predictions

def evaluate_forecast(test_data, predictions):
    df_new = pd.DataFrame({'Test': test_data, 'Predictions': predictions})
    df_new['Date'] = df['Date'].tail(len(test_data)).reset_index(drop=True)
    df_new['Date'] = pd.to_datetime(df_new['Date'])

    plt.figure(figsize=(20, 13))
    ax = plt.gca()
    df_new.Test.plot(color='blue', label='Real Price')
    df_new.Predictions.plot(color='r', ls='--', label='Predicted Weighted_Price')
    plt.legend()
    plt.ylabel('Price')
    plt.title('BTC Price Prediction')
    plt.show()

    mae = np.mean(np.abs(df_new['Test'] - df_new['Predictions']))
    mape = np.mean(np.abs((df_new['Test'] - df_new['Predictions']) / df_new['Test'])) * 100
    rmse = np.sqrt(np.mean((df_new['Test'] - df_new['Predictions'])**2))
    r_squared = r2_score(df_new['Test'], df_new['Predictions'])

    metrics_df = pd.DataFrame({
        'MAE': [mae],
        'MAPE': [mape],
        'RMSE': [rmse],
        'R^2': [r_squared]
    })

    print(metrics_df)

def main():
    warnings.filterwarnings('ignore')

    file_path = 'BTC-USD.csv'
	 data = load_data(file_path)
    data = make_data_stationary(data)

    train_data, test_data = train_test_split_data(data)

    predictions = fit_arima_model(train_data)

    evaluate_forecast(test_data, predictions)

if __name__ == "__main__":
    main()
