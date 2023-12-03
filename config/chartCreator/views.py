import numpy as np
from django.http import JsonResponse
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import requests
import json
from django.conf import settings
import os
# from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from scipy.stats import t


def split_data(data, test_size=0.2):
    split_idx = int(len(data) * (1 - test_size))
    train, test = data[:split_idx], data[split_idx:]
    return train, test


def compute_RSI(series, period=14):
    delta = series.diff().dropna()
    loss = delta.where(delta < 0, 0)
    gain = -delta.where(delta > 0, 0)
    avg_loss = loss.rolling(window=period).mean()
    avg_gain = gain.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_MACD(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def compute_RMSE(y_true, y_pred):
    differences = [true - pred for true, pred in zip(y_true, y_pred)]
    return np.sqrt(sum([diff ** 2 for diff in differences]) / len(differences))


def compute_MAE(y_true, y_pred):
    differences = [true - pred for true, pred in zip(y_true, y_pred)]
    return sum([abs(diff) for diff in differences]) / len(differences)


def transform_data_for_cnn(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Вхідні дані для CNN повинні мати форму [samples, timesteps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    return X, y


def train_cnn_model(X, y, epochs=50, batch_size=32):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X.shape[1], X.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    return model


def prepare_data(data):
    df = pd.DataFrame(data)

    # Saving original dates
    original_dates = df['time_period_start'].copy()

    # Calculate technical indicators
    df['RSI'] = compute_RSI(df['price_close'])
    macd, signal = compute_MACD(df['price_close'])
    df['MACD'] = macd
    df['Signal'] = signal

    # Calculate other features such as volatility
    df['Volatility'] = df['price_high'] - df['price_low']

    # Remove unnecessary columns
    columns_to_remove = ['time_period_end', 'time_open', 'time_close', 'time_period_start']
    df.drop(columns_to_remove, axis=1, inplace=True)

    return df, original_dates


def transform_data_for_lstm(data, lag=1):
    df = pd.DataFrame(data)

    # Масштабування даних
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df.values.reshape(-1, 1))

    df_scaled = pd.DataFrame(df_scaled)
    columns = [df_scaled.shift(i) for i in range(1, lag + 1)]
    columns.append(df_scaled)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Вхідні дані для LSTM повинні мати форму [samples, timesteps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    return X, y, scaler


def train_lstm_model(X, y, epochs=50, batch_size=32, neurons=50):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))  # Додатковий LSTM шар
    model.add(Dropout(0.2))  # Dropout шар для запобігання перенавчанню
    model.add(LSTM(neurons))
    model.add(Dropout(0.2))  # Dropout шар
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Рання зупинка
    es = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.2,
              callbacks=[es])  # 20% даних використовуються для валідації

    return model


def lstm_forecast(model, data, scaler):
    data = data.reshape(data.shape[0], 1, data.shape[1])
    forecast = model.predict(data)

    # Перетворення прогнозування назад до оригінального масштабу
    forecast = scaler.inverse_transform(forecast)
    return forecast


def forecast_prices(df, forecast_periods=30):
    print('testas_flag_for_print --', df.dtypes)

    # Вибираємо значення закриття для прогнозування
    price_close = df['price_close']

    if price_close.isnull().any():
        raise ValueError("price_close contains NaN values!")

    # Спробуємо автоматичний вибір порядку за допомогою pmdarima
    # ARIMA прогноз
    try:
        model_arima = auto_arima(price_close, seasonal=True, trace=False, m=12)
        forecast_arima = model_arima.predict(n_periods=forecast_periods)
    except Exception as e:
        print('Error during ARIMA forecasting:', str(e))
        raise ValueError(f"Error forecasting with ARIMA model: {str(e)}")

    # LSTM прогноз
    data_lstm_X, data_lstm_y, scaler = transform_data_for_lstm(price_close.values)
    model_lstm = train_lstm_model(data_lstm_X, data_lstm_y)
    forecast_lstm = lstm_forecast(model_lstm, data_lstm_X[-forecast_periods:], scaler)

    # Навчання моделі Random Forest:
    X = df.drop(['price_close'], axis=1)
    y = df['price_close']
    df.fillna(df.mean(), inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    # Навчання моделі Gradient Boosting:
    gb_model = GradientBoostingRegressor(n_estimators=100)
    gb_model.fit(X_train, y_train)
    gb_predictions = gb_model.predict(X_test)

    # CNN прогноз
    X_cnn, y_cnn = transform_data_for_cnn(price_close.values, lag=5)
    cnn_model = train_cnn_model(X_cnn, y_cnn)
    cnn_predictions = cnn_model.predict(X_cnn[-forecast_periods:]).flatten()

    # Об'єднання прогнозів:
    combined_forecast = (np.array(forecast_arima) + np.array(
        forecast_lstm.flatten()) + rf_predictions + gb_predictions + cnn_predictions) / 5

    return combined_forecast


def stack_models(df, forecast_periods=30):
    """
    Функція для стекінга різних моделей для прогнозування часових рядів.

    Параметри:
        df (pd.DataFrame): вхідний датафрейм з історією часових рядів
        forecast_periods (int): кількість періодів для прогнозування в майбутнє

    Повертає:
        np.array: прогнозовані значення для заданої кількості періодів
    """

    # Розділення даних на навчальні та тестові за допомогою split_data
    train_df, test_df = split_data(df)
    X_train, y_train = train_df.drop(['price_close'], axis=1), train_df['price_close']
    X_test, y_test = test_df.drop(['price_close'], axis=1), test_df['price_close']

    # Виправимо для прогнозів RF та GB
    X_all = df.drop(['price_close'], axis=1)

    # Прогнозування ARIMA
    model_arima = auto_arima(y_train, seasonal=True, trace=False, m=12)
    predictions_arima = model_arima.predict(n_periods=len(y_test))
    future_arima = model_arima.predict(n_periods=forecast_periods)

    # Обчислення RMSE та MAE для ARIMA прогнозу
    rmse_arima = compute_RMSE(y_test, predictions_arima)
    mae_arima = compute_MAE(y_test, future_arima)
    print("ARIMA RMSE:", rmse_arima)
    print("ARIMA MAE:", mae_arima)

    # Прогнозування LSTM
    lstm_train_data_X, lstm_train_data_y, lstm_train_scaler = transform_data_for_lstm(y_train)
    if lstm_train_data_X.shape[0] == 0:
        raise ValueError("LSTM training data is empty or improperly formatted!")

    model_lstm = train_lstm_model(lstm_train_data_X, lstm_train_data_y)

    lstm_test_data_X, lstm_test_data_y, lstm_test_scaler = transform_data_for_lstm(y_test)
    if lstm_test_data_X.shape[0] == 0:
        raise ValueError("LSTM test data is empty or improperly formatted!")
    predictions_lstm = lstm_forecast(model_lstm, lstm_test_data_X, lstm_test_scaler).flatten()

    if predictions_lstm.shape[0] != y_test.shape[0]:
        raise ValueError(
            f"LSTM predictions length ({predictions_lstm.shape[0]}) doesn't match y_test length ({y_test.shape[0]})!")

    lstm_future_data_X, lstm_future_data_y, lstm_future_scaler = transform_data_for_lstm(df['price_close'])
    if len(lstm_future_data_X) < forecast_periods:
        raise ValueError(f"LSTM future data has less data than forecast_periods ({forecast_periods})!")

    future_lstm_input = lstm_future_data_X[-forecast_periods:]
    future_lstm = lstm_forecast(model_lstm, future_lstm_input, lstm_future_scaler).flatten()

    if future_lstm.shape[0] != forecast_periods:
        raise ValueError(
            f"LSTM future forecast length ({future_lstm.shape[0]}) doesn't match forecast_periods ({forecast_periods})!")

    # Обчислення RMSE та MAE для LSTM прогнозу
    rmse_lstm = compute_RMSE(y_test, predictions_lstm)
    mae_lstm = compute_MAE(y_test, predictions_lstm)
    print("LSTM RMSE:", rmse_lstm)
    print("LSTM MAE:", mae_lstm)

    # Прогнозування RF
    model_rf = RandomForestRegressor(n_estimators=100)
    model_rf.fit(X_train, y_train)
    predictions_rf = model_rf.predict(X_test)
    future_rf = model_rf.predict(X_all.iloc[-forecast_periods:])

    # Обчислення RMSE та MAE для RF прогнозу
    rmse_rf = compute_RMSE(y_test, predictions_rf)
    mae_rf = compute_MAE(y_test, future_rf)
    print("RF RMSE:", rmse_rf)
    print("RF MAE:", mae_rf)

    # Прогнозування GB
    model_gb = GradientBoostingRegressor(n_estimators=100)
    model_gb.fit(X_train, y_train)
    predictions_gb = model_gb.predict(X_test)
    future_gb = model_gb.predict(X_all.iloc[-forecast_periods:])

    # Обчислення RMSE та MAE для GB прогнозу
    rmse_gb = compute_RMSE(y_test, predictions_gb)
    mae_gb = compute_MAE(y_test, future_gb)
    print("RF RMSE:", rmse_gb)
    print("RF MAE:", mae_gb)

    # Прогнозування CNN
    X_cnn, y_cnn = transform_data_for_cnn(y_train.values, lag=5)
    cnn_model = train_cnn_model(X_cnn, y_cnn)
    predictions_cnn = cnn_model.predict(transform_data_for_cnn(y_test.values, lag=5)[0]).flatten()
    future_cnn = cnn_model.predict(
        transform_data_for_cnn(df['price_close'].values, lag=5)[0][-forecast_periods:]).flatten()

    # Обчислення RMSE та MAE для CNN прогнозу
    rmse_cnn = compute_RMSE(y_test, predictions_cnn)
    mae_cnn = compute_MAE(y_test, future_cnn)
    print("CNN RMSE:", rmse_cnn)
    print("CNN MAE:", mae_cnn)

    # Стекінг прогнозів для мета-моделі
    stacked_features = np.column_stack(
        [predictions_arima, predictions_lstm, predictions_rf, predictions_gb, predictions_cnn])

    meta_model = LinearRegression()
    meta_model.fit(stacked_features, y_test)

    future_stacked_features = np.column_stack([future_arima, future_lstm, future_rf, future_gb, future_cnn])
    future_predictions = meta_model.predict(future_stacked_features)

    return future_predictions


def calculate_returns(df):
    df['returns'] = df['price_close'].pct_change()
    return df


def calculate_future_returns(forecasted_prices):
    returns = np.diff(forecasted_prices) / forecasted_prices[:-1]
    return returns


def calculate_var(returns, confidence_level=0.95):
    if len(returns) == 0:
        return None
    # Сортування доходностей
    sorted_returns = np.sort(returns)
    # Розрахунок VaR
    index = int((1 - confidence_level) * len(sorted_returns))
    var = sorted_returns[index]
    return var


def calculate_historical_var(returns, confidence_level=0.95):
    sorted_returns = np.sort(returns)
    index = int(confidence_level * len(sorted_returns))
    var = -sorted_returns[index]
    return var


def calculate_t_dist_es(returns, confidence_level=0.95):
    dof, mean, scale = t.fit(returns)
    es = t.expect(lambda x: x, args=(dof,), loc=mean, scale=scale, lb=mean - scale * t.ppf(confidence_level, dof))
    return es


def calculate_es(returns, confidence_level=0.95):
    if len(returns) == 0:
        return None

    # Сортування доходностей
    sorted_returns = np.sort(returns)
    # Розрахунок VaR
    index = int((1 - confidence_level) * len(sorted_returns))
    var = sorted_returns[index]
    # ES - середня втрата у найгірших (1-confidence_level)% випадках
    es = sorted_returns[:index].mean()

    return es


def get_historical_data(request, name, count, start_time, end_time=None):
    if not end_time:
        from datetime import datetime
        end_time = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

    url = f'https://rest.coinapi.io/v1/ohlcv/{name}/history?period_id=1DAY&time_start={start_time}&time_end={end_time}&limit={count}'

    headers = {
        'X-CoinAPI-Key': 'FF4AACC3-C6FF-47A1-8F4D-5EB8DC574699'
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        # Обробка даних і збереження оригінальних дат
        df, original_dates = prepare_data(data)
        df = calculate_returns(df)  # Розрахунок доходностей
        df.fillna(df.mean(), inplace=True)

        if 'price_close' not in df.columns:
            return JsonResponse({"success": False, "response": "price_close column not found in DataFrame"}, status=400)

        # Нормалізація та кластеризація
        try:
            scaler = MinMaxScaler()
            df_normalized = pd.DataFrame(scaler.fit_transform(df[['price_close']]))
            kmeans = KMeans(n_clusters=3)
            kmeans.fit(df_normalized)
            df['cluster'] = kmeans.predict(df_normalized)
            distances = kmeans.transform(df_normalized)
            df['risk'] = distances.min(axis=1)
        except Exception as e:
            return JsonResponse({"success": False, "response": f"Error during data processing: {str(e)}"}, status=500)

        # Прогнозування майбутніх цін
        try:
            future_prices = stack_models(df)

            # Розрахунок майбутніх доходностей
            future_returns = calculate_future_returns(future_prices)

            # Розрахунок VaR та ES
            var = calculate_historical_var(future_returns)
            es = calculate_t_dist_es(future_returns)

            # Додавання оригінальних дат до результату
            historical_data_with_dates = df.assign(date=original_dates).to_dict(orient="records")
        except Exception as e:
            return JsonResponse({"success": False, "response": f"Error during forecasting: {str(e)}"}, status=500)

        response_data = {
            "historical_data": historical_data_with_dates,
            "forecast": future_prices.tolist(),
            "VaR": var,
            "ES": es
        }

        return JsonResponse({"success": True, "response": response_data}, status=200)
    else:
        error_message = response.json().get('error', 'Unknown error')
        return JsonResponse({"success": False, "response": f"Error: {response.status_code}. Message: {error_message}"},
                            status=response.status_code)


def load_data_from_json():
    file_path = os.path.join(settings.BASE_DIR, 'chartCreator', 'response.json')
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def create_chart(request, name, start_time, end_time):
    return JsonResponse({"success": True, "name": name, "startTime": start_time, "endTime": end_time}, status=200)


def get_symbols(request):
    url = 'https://rest.coinapi.io/v1/symbols'

    headers = {
        'X-CoinAPI-Key': 'FF4AACC3-C6FF-47A1-8F4D-5EB8DC574699'
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        symbols = [item['symbol_id'] for item in data]
        return JsonResponse({"success": True, "response": symbols}, status=200)
    else:
        error_message = response.json().get('error', 'Unknown error')
        return JsonResponse({"success": False, "response": f"Error: {response.status_code}. Message: {error_message}"},
                            status=response.status_code)
