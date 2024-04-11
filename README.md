# Predict__BTC__Price
with me and my friend you can do everything!


Here is the code for u!

            import yfinance as yf
            import numpy as np
            import pandas as pd
            from sklearn.preprocessing import MinMaxScaler
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from sklearn.metrics import mean_absolute_error
            from typing import List, Tuple
            def create_dataset(data: List[float], lookback: int, future_steps: int) -> Tuple[np.ndarray, np.ndarray]:
               from typing import List, Tuple
            def create_dataset(data: List[float], lookback: int, future_steps: int) -> Tuple[np.ndarray, np.ndarray]:
                X = []
                Y = []
            
                for i in range(len(data) - lookback - future_steps - 1):
                    X.append(data[i:(i + lookback)])
                    Y.append(data[(i + lookback):(i + lookback + future_steps)])
            
                X = np.array(X)
                Y = np.array(Y)
            
                return X, Y
            
            
            
            
            
            data = yf.download(tickers='BTC-USD', period='60h', interval='5m')
            data = data[['Close']]
            
            
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data)
            
            
            train_size = int(len(data_scaled) * 0.7)
            train_data = data_scaled[:train_size]
            test_data = data_scaled[train_size:]
            
            
            lookback = 50
            future_steps = 60*30
            X_train, Y_train = create_dataset(train_data, lookback, future_steps)
            X_test, Y_test = create_dataset(test_data, lookback, future_steps)
            
            lookback = 6
            future_steps = 1
            X_train, Y_train = create_dataset(train_data, lookback, future_steps)
            X_test, Y_test = create_dataset(test_data, lookback, future_steps)
            
            
            model = Sequential()
            model.add(LSTM(128, return_sequences=True, input_shape=(lookback, 1)))
            model.add(LSTM(64))
            model.add(Dense(future_steps))
            
            
            model.compile(optimizer='rmsprop', loss='mse')
            model.fit(X_train, Y_train, epochs=50, batch_size=128)
            
            predictions = model.predict(X_test)
            
            
            Y_test = np.reshape(Y_test, (Y_test.shape[0], Y_test.shape[1]))
            predictions = np.reshape(predictions, (predictions.shape[0], predictions.shape[1]))
            
            
            Y_test = scaler.inverse_transform(Y_test)
            predictions = scaler.inverse_transform(predictions)
            
            last_10_predictions = predictions[-10:]
            average_predicted_price = np.mean(last_10_predictions)
            print("پیش‌بینی قیمت بیتکوین بر اساس میانگین آخرین 10 پیش‌بینی:")
            print(average_predicted_price)
            
            last_observed_price = test_data[-1][-1]
            
            price_difference = abs(predictions[-1][-1] - last_observed_price)
            
            threshold = 0.1
