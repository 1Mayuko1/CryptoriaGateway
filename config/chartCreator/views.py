import numpy as np
from django.http import JsonResponse
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import requests
import json
from django.conf import settings
import os
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from keras.models import Sequential
from keras.layers import LSTM, Dense


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
    # Вибираємо значення закриття для прогнозування
    df.set_index('time_period_start', inplace=True)
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

    # Комбінований прогноз
    forecast_combined = (np.array(forecast_arima) + np.array(forecast_lstm.flatten())) / 2

    return forecast_combined


# def forecast_prices(df, forecast_periods=30):
#     # Вибираємо значення закриття для прогнозування
#     df.set_index('time_period_start', inplace=True)
#     price_close = df['price_close']
#
#     if price_close.isnull().any():
#         raise ValueError("price_close contains NaN values!")
#
#     print('aloaloaloalo - ', price_close.index)
#
#     # Створюємо модель ARIMA
#     try:
#         model = ARIMA(price_close, order=(5, 1, 0))  # order=(p,d,q)
#     except Exception as e:
#         raise ValueError(f"Error creating ARIMA model: {str(e)}")
#
#     # Застосовуємо модель
#     try:
#         model_fit = model.fit()
#     except Exception as e:
#         raise ValueError(f"Error fitting ARIMA model: {str(e)}")
#
#     # Виконуємо прогноз
#     try:
#         forecast = model_fit.forecast(steps=forecast_periods)[0]
#     except Exception as e:
#         print('dodomy TESTAS', e)
#         raise ValueError(f"Error forecasting with ARIMA model: {str(e)}")
#
#     return forecast


def load_data_from_json():
    file_path = os.path.join(settings.BASE_DIR, 'chartCreator', 'response.json')
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def create_chart(request, name, start_time, end_time):
    return JsonResponse({"success": True, "name": name, "startTime": start_time, "endTime": end_time}, status=200)


# def get_historical_data(request, name, count, start_time, end_time=None):
#     if not end_time:
#         from datetime import datetime
#         end_time = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
#
#     url = f'https://rest.coinapi.io/v1/ohlcv/{name}/history?period_id=1DAY&time_start={start_time}&time_end={end_time}&limit={count}'
#
#     headers = {
#         'X-CoinAPI-Key': 'FF4AACC3-C6FF-47A1-8F4D-5EB8DC574699'
#     }
#
#     response = requests.get(url, headers=headers)
#
#     if response.status_code == 200:
#         data = response.json()
#         return JsonResponse({"success": True, "response": data}, status=200)
#     else:
#         error_message = response.json().get('error', 'Unknown error')
#         return JsonResponse({"success": False, "response": f"Error: {response.status_code}. Message: {error_message}"},
#                             status=response.status_code)

# def get_historical_data(request, name, count, start_time, end_time=None):
#     if not end_time:
#         from datetime import datetime
#         end_time = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
#
#     # Завантажуємо дані з файлу замість API-запиту
#     data = load_data_from_json()
#
#     # Конвертуємо дані в DataFrame
#     df = pd.DataFrame(data["response"])
#
#     # Нормалізація даних
#     scaler = MinMaxScaler()
#     df_normalized = pd.DataFrame(scaler.fit_transform(df[['price_close']]))
#
#     # Використання KMeans для кластеризації
#     kmeans = KMeans(n_clusters=3)  # Можемо змінити кількість кластерів
#     kmeans.fit(df_normalized)
#     df['cluster'] = kmeans.predict(df_normalized)
#
#     # Обчислення відстані до центрів кластерів (це буде наш ризик)
#     distances = kmeans.transform(df_normalized)
#     df['risk'] = distances.min(axis=1)
#
#     # Тепер повертаємо оброблені дані
#     return JsonResponse({"success": True, "response": df.to_dict(orient="records")}, status=200)

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

        # Конвертуємо дані в DataFrame
        df = pd.DataFrame(data)

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
            future_prices = forecast_prices(df)
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
