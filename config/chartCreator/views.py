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


def prepare_data(data):
    df = pd.DataFrame(data)

    # Обчислення технічних показників
    df['RSI'] = compute_RSI(df['price_close'])
    macd, signal = compute_MACD(df['price_close'])
    df['MACD'] = macd
    df['Signal'] = signal

    # Обчислення інших характеристик, як-от волатильність
    df['Volatility'] = df['price_high'] - df['price_low']

    # Обчислення обсягу торгів
    # Ви вже маєте 'volume_traded' в ваших даних, тому просто переконайтеся, що вони відсортовані вірно.

    # Видалення непотрібних стовпців
    columns_to_remove = ['time_period_end', 'time_open', 'time_close', 'time_period_start']
    df.drop(columns_to_remove, axis=1, inplace=True)

    return df


def transform_data_for_lstm(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Вхідні дані для LSTM повинні мати форму [samples, timesteps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    return X, y


def train_lstm_model(X, y, epochs=50, batch_size=32, neurons=50):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    return model


def lstm_forecast(model, data):
    data = data.reshape(data.shape[0], 1, data.shape[1])
    return model.predict(data)


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
    data_lstm_X, data_lstm_y = transform_data_for_lstm(price_close.values)
    model_lstm = train_lstm_model(data_lstm_X, data_lstm_y)
    forecast_lstm = lstm_forecast(model_lstm, data_lstm_X[-forecast_periods:])

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

    # Об'єднання прогнозів:
    combined_forecast = (np.array(forecast_arima) + np.array(
        forecast_lstm.flatten()) + rf_predictions + gb_predictions) / 4

    return combined_forecast


# ... ваші інші функції ...

def stack_models(df, forecast_periods=30):

    # Дані для навчання та тестування
    X = df.drop(['price_close'], axis=1)
    y = df['price_close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Отримання прогнозів для ARIMA моделі
    model_arima = auto_arima(y_train, seasonal=True, trace=False, m=12)
    predictions_arima = model_arima.predict(n_periods=len(y_test))
    future_arima = model_arima.predict(n_periods=forecast_periods)

    # Отримання прогнозів для LSTM моделі
    model_lstm = train_lstm_model(transform_data_for_lstm(y_train)[0], transform_data_for_lstm(y_train)[1])

    lstm_data = transform_data_for_lstm(y_test)[0][-forecast_periods:]
    if lstm_data.shape[0] == 0:
        raise ValueError("LSTM data for predictions is empty!")
    predictions_lstm = lstm_forecast(model_lstm, transform_data_for_lstm(y_test)[0]).flatten()

    lstm_future_data = transform_data_for_lstm(y.values)[0][-forecast_periods:]
    if lstm_future_data.shape[0] == 0:
        raise ValueError("LSTM data for future forecasting is empty!")
    future_lstm = lstm_forecast(model_lstm, lstm_future_data).flatten()

    # Отримання прогнозів для RF моделі
    model_rf = RandomForestRegressor(n_estimators=100)
    model_rf.fit(X_train, y_train)
    predictions_rf = model_rf.predict(X_test)
    future_rf = model_rf.predict(X.iloc[-forecast_periods:])

    # Отримання прогнозів для GB моделі
    model_gb = GradientBoostingRegressor(n_estimators=100)
    model_gb.fit(X_train, y_train)
    predictions_gb = model_gb.predict(X_test)
    future_gb = model_gb.predict(X.iloc[-forecast_periods:])

    # Використання прогнозів як особливостей для мета-моделі
    # print("predictions_arima:", len(predictions_arima))
    # print("predictions_lstm:", len(predictions_lstm))
    # print("predictions_rf:", len(predictions_rf))
    # print("predictions_gb:", len(predictions_gb))

    stacked_features = np.column_stack([predictions_arima, predictions_lstm, predictions_rf, predictions_gb])

    # Навчання мета-моделі
    meta_model = LinearRegression()
    meta_model.fit(stacked_features, y_test)

    # Прогнозування майбутніх цін
    # print("future_arima:", len(future_arima))
    # print("future_lstm:", len(future_lstm))
    # print("future_rf:", len(future_rf))
    # print("future_gb:", len(future_gb))
    future_stacked_features = np.column_stack([future_arima, future_lstm, future_rf, future_gb])

    future_predictions = meta_model.predict(future_stacked_features)

    return future_predictions


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

        # # Конвертуємо дані в DataFrame
        # df = pd.DataFrame(data)

        # Нова підготовка даних
        df = prepare_data(data)

        # Заповнюємо NaN значення середніми значеннями
        df.fillna(df.mean(), inplace=True)

        if 'price_close' not in df.columns:
            return JsonResponse({"success": False, "response": "price_close column not found in DataFrame"}, status=400)

        # Нормалізація даних
        try:
            scaler = MinMaxScaler()
            df_normalized = pd.DataFrame(scaler.fit_transform(df[['price_close']]))
        except Exception as e:
            return JsonResponse({"success": False, "response": f"Error during normalization: {str(e)}"}, status=500)

        try:
            # Використання KMeans для кластеризації
            kmeans = KMeans(n_clusters=3)
            kmeans.fit(df_normalized)
            df['cluster'] = kmeans.predict(df_normalized)
            # Обчислення відстані до центрів кластерів (це буде наш ризик)
            distances = kmeans.transform(df_normalized)
            df['risk'] = distances.min(axis=1)
        except Exception as e:
            return JsonResponse({"success": False, "response": f"Error during KMeans clustering: {str(e)}"}, status=500)

        # Прогнозуємо майбутні ціни
        try:
            future_prices = stack_models(df)
        except Exception as e:
            return JsonResponse({"success": False, "response": f"Error during forecasting: {str(e)}"}, status=500)

        # Додаємо прогнозовані ціни до відповіді
        response_data = {
            "historical_data": df.to_dict(orient="records"),
            "forecast": future_prices.tolist()
        }

        # Тепер повертаємо оброблені дані
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
