import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(['Adj Close', 'Volume'], axis=1)
    df = df.iloc[:, 1:]
    min_max_scaler = preprocessing.MinMaxScaler()
    df_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(df_scaled, columns=df.columns)
    return df_normalized

def create_sequences(data, window):
    sequence_length = window + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    result = np.array(result)
    return result

def prepare_data(data, window):
    X = data[:, :-1]
    y = data[:, -1][:, -1]
    X = np.reshape(X, (X.shape[0], X.shape[1], len(data[0][0])))
    return X, y

def build_model(window, input_size, lstm_units, dropout):
    inputs = Input(shape=(window, input_size))
    model = Conv1D(filters=lstm_units, kernel_size=1, activation='sigmoid')(inputs)
    model = MaxPooling1D(pool_size=window)(model)
    model = Dropout(dropout)(model)
    model = Bidirectional(LSTM(lstm_units, activation='tanh'), name='bilstm')(model)
    attention = Dense(lstm_units * 2, activation='sigmoid', name='attention_vec')(model)
    model = Multiply()([model, attention])
    outputs = Dense(1, activation='tanh')(model)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model

def plot_predictions(df_combined):
    plt.figure(figsize=(20, 13))
    df_combined.real.plot()
    df_combined.predict.plot(color='r', ls='--')
    plt.legend()
    plt.title('Test Data')
    plt.xlabel('Date')
    plt.ylabel('mean USD')
    plt.show()

def calculate_metrics(df_combined):
    mae = np.mean(np.abs(df_combined['real'] - df_combined['predict']))
    mape = np.mean(np.abs((df_combined['real'] - df_combined['predict']) / df_combined['real'])) * 100
    rmse = np.sqrt(np.mean((df_combined['real'] - df_combined['predict'])**2))
    r_squared = r2_score(df_combined['real'], df_combined['predict'])

    metrics_df = pd.DataFrame({
        'MAE': [mae],
        'MAPE': [mape],
        'RMSE': [rmse],
        'R^2': [r_squared]
    })

    return metrics_df

if __name__ == "__main__":
    # Load and preprocess data
    window = 5
    lstm_units = 16
    dropout = 0.01
    epoch = 100

    file_path = 'BTC-USD.csv'
    df = load_data(file_path)

    # Prepare data for LSTM model
    data = create_sequences(df.values, window)
    X_train, y_train = prepare_data(data[:int(0.7 * len(data))], window)
    X_test, y_test = prepare_data(data[int(0.7 * len(data)):], window)

    # Build and train the LSTM model
    input_size = X_train.shape[2]
    model = build_model(window, input_size, lstm_units, dropout)
    model.summary()
    history = model.fit(X_train, y_train, epochs=epoch, batch_size=256, shuffle=False, validation_data=(X_test, y_test))

    # Predictions and evaluation
    y_test_predict = model.predict(X_test)
    df_combined = pd.DataFrame({
        "Date": df.iloc[int(0.7 * len(data)) + window:, 0],
        "predict": y_test_predict.flatten(),
        "real": y_test.flatten()
    })
    df_combined["Date"] = pd.to_datetime(df_combined["Date"])

    # Plot predictions and calculate metrics
    plot_predictions(df_combined)
    metrics_df = calculate_metrics(df_combined)
    print(metrics_df)
