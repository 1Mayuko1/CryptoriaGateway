import numpy as np
from django.http import JsonResponse
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import requests
import json
from django.conf import settings
import os
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
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import levene
from joblib import dump, load
import os


def save_model(model, filename):
    dump(model, filename)


def load_model(filename):
    return load(filename)


def train_or_load_model(model_name, train_function, *args, **kwargs):
    model_file = f'{model_name}.joblib'

    if os.path.exists(model_file):
        print(f"Завантаження моделі {model_name}")
        model = load_model(model_file)
    else:
        print(f"Навчання моделі {model_name}")
        model = train_function(*args, **kwargs)
        save_model(model, model_file)

    return model


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
    # model.add(MaxPooling1D(pool_size=2)) # Видаліть або змініть цей шар
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
        dict: прогнозовані значення та метрики
    """
    try:
        model_efficiency = {}
        models_used = []
        rmse_values = {}
        mae_values = {}
        all_models = ['ARIMA', 'LSTM', 'RandomForest', 'GradientBoosting', 'CNN']

        model_scores = cross_validate_models(df)
        best_models = determine_best_model_combination(model_scores)
        models_used.extend(best_models)

        total_error = sum(model_scores.values())
        for model, error in model_scores.items():
            model_efficiency[model] = ((total_error - error) / total_error) * 100

        train_df, test_df = split_data(df)
        X_train, y_train = train_df.drop(['price_close'], axis=1), train_df['price_close']
        X_test, y_test = test_df.drop(['price_close'], axis=1), test_df['price_close']

        # Отримання прогнозів від різних моделей
        predictions = []
        for model in best_models:
            pred, rmse, mae = perform_prediction_and_metrics(model, X_train, X_test, y_train, y_test, df)
            if pred is not None:
                predictions.append(pred)
            rmse_values[model] = rmse
            mae_values[model] = mae

        # Обрізка прогнозів до мінімальної довжини
        min_length = min(len(pred) for pred in predictions)
        trimmed_predictions = [pred[:min_length] for pred in predictions]

        # Об'єднання прогнозів
        if trimmed_predictions:
            stacked_predictions = np.column_stack(trimmed_predictions)
            meta_model = LinearRegression()
            meta_model.fit(stacked_predictions, y_test[:min_length])
            forecast = meta_model.predict(stacked_predictions)
            stacking_rmse = compute_RMSE(y_test[:min_length], forecast)

        average_rmse = np.mean(list(rmse_values.values()))
        stackingEfficiency = 100 * (average_rmse - stacking_rmse) / average_rmse if average_rmse else 0

        return {
            "forecast": forecast.tolist() if forecast is not None else "No forecast",
            "rmse": rmse_values,
            "mae": mae_values,
            "modelsUsed": models_used,
            "models": all_models,
            "modelsEfficiency": model_efficiency,
            "stackingEfficiency": stackingEfficiency,
        }

    except Exception as e:
        print("An error occurred during model stacking: ", str(e))
        return None


def perform_prediction_and_metrics(model_name, X_train, X_test, y_train, y_test, df):
    try:
        predictions, rmse, mae = None, None, None

        if model_name == 'ARIMA':
            model_arima = train_or_load_model('arima', auto_arima, y_train, seasonal=True, trace=False, m=12)
            predictions = model_arima.predict(n_periods=len(y_test))

        elif model_name == 'LSTM':
            lstm_train_data_X, lstm_train_data_y, lstm_train_scaler = transform_data_for_lstm(y_train)
            if lstm_train_data_X.size == 0 or lstm_train_data_y.size == 0:
                raise ValueError("Empty training data for LSTM model.")
            model_lstm = train_or_load_model('lstm', train_lstm_model, lstm_train_data_X, lstm_train_data_y)
            predictions = lstm_forecast(model_lstm, lstm_train_data_X, lstm_train_scaler).flatten()

        elif model_name == 'RandomForest':
            model_rf = RandomForestRegressor(n_estimators=100)
            model_rf = train_or_load_model('random_forest', lambda X, y: model_rf.fit(X, y), X_train, y_train)
            predictions = model_rf.predict(X_test)

        elif model_name == 'GradientBoosting':
            model_gb = GradientBoostingRegressor(n_estimators=100)
            model_gb = train_or_load_model('gradient_boosting', lambda X, y: model_gb.fit(X, y), X_train, y_train)
            predictions = model_gb.predict(X_test)

        elif model_name == 'CNN':
            X_cnn, y_cnn = transform_data_for_cnn(y_train.values, lag=5)
            if X_cnn.size == 0 or y_cnn.size == 0:
                raise ValueError("Empty training data for CNN model.")
            model_cnn = train_or_load_model('cnn', train_cnn_model, X_cnn, y_cnn)
            predictions = model_cnn.predict(transform_data_for_cnn(y_test.values, lag=5)[0]).flatten()

        else:
            raise ValueError(f"Model {model_name} is not recognized.")

        if predictions is not None:
            rmse = compute_RMSE(y_test, predictions)
            mae = compute_MAE(y_test, predictions)

        return predictions, rmse, mae

    except Exception as e:
        print(f"Error in model {model_name}: {str(e)}")
        return None, None, None


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


def advanced_data_analysis(data):
    results = {}

    # Аналіз стационарності
    adf_test = adfuller(data['price_close'])
    kpss_test = kpss(data['price_close'], nlags='auto')
    results['stationary'] = {
        'adf_statistic': adf_test[0], 'adf_pvalue': adf_test[1],
        'kpss_statistic': kpss_test[0], 'kpss_pvalue': kpss_test[1]
    }

    # Сезонність
    decomposed = seasonal_decompose(data['price_close'], model='additive', period=30)
    results['seasonality'] = {
        'seasonal': decomposed.seasonal,
        'trend': decomposed.trend,
        'resid': decomposed.resid,
        'seasonal_strength': np.std(decomposed.seasonal) / np.std(data['price_close'])
    }

    # Автокореляція
    results['acf_values'] = acf(data['price_close'])
    results['pacf_values'] = pacf(data['price_close'])

    # Волатильність (Використання Levene Test для перевірки однорідності волатильності)
    levene_test = levene(data['price_high'], data['price_low'])
    results['volatility'] = {
        'levene_statistic': levene_test.statistic,
        'levene_pvalue': levene_test.pvalue
    }

    return results


def cross_validate_models(data, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    errors = {'ARIMA': [], 'LSTM': [], 'RandomForest': [], 'GradientBoosting': [], 'CNN': []}

    for train_idx, test_idx in tscv.split(data):
        train, test = data.iloc[train_idx], data.iloc[test_idx]

        # ARIMA
        model_arima = train_or_load_model('arima_cv', auto_arima, train['price_close'], seasonal=True, trace=False, m=12)
        predictions_arima = model_arima.predict(n_periods=len(test))
        errors['ARIMA'].append(mean_squared_error(test['price_close'], predictions_arima))

        # LSTM
        X_lstm, y_lstm, scaler_lstm = transform_data_for_lstm(train['price_close'])
        lstm_model = train_or_load_model('lstm_cv', train_lstm_model, X_lstm, y_lstm)
        X_test_lstm, y_test_lstm, _ = transform_data_for_lstm(test['price_close'])
        lstm_predictions = lstm_forecast(lstm_model, X_test_lstm, scaler_lstm)
        errors['LSTM'].append(mean_squared_error(test['price_close'], lstm_predictions.flatten()))

        # RandomForest
        rf_model = RandomForestRegressor(n_estimators=100)
        rf_model = train_or_load_model('random_forest_cv', lambda X, y: rf_model.fit(X, y), train.drop('price_close', axis=1), train['price_close'])
        rf_predictions = rf_model.predict(test.drop('price_close', axis=1))
        errors['RandomForest'].append(mean_squared_error(test['price_close'], rf_predictions))

        # GradientBoosting
        gb_model = GradientBoostingRegressor(n_estimators=100)
        gb_model = train_or_load_model('gradient_boosting_cv', lambda X, y: gb_model.fit(X, y), train.drop('price_close', axis=1), train['price_close'])
        gb_predictions = gb_model.predict(test.drop('price_close', axis=1))
        errors['GradientBoosting'].append(mean_squared_error(test['price_close'], gb_predictions))

        # CNN
        X_cnn, y_cnn = transform_data_for_cnn(train['price_close'], lag=2)
        cnn_model = train_or_load_model('cnn_cv', train_cnn_model, X_cnn, y_cnn)
        X_test_cnn, y_test_cnn = transform_data_for_cnn(test['price_close'], lag=2)[0:2]
        cnn_predictions = cnn_model.predict(X_test_cnn).flatten()
        errors['CNN'].append(mean_squared_error(y_test_cnn, cnn_predictions))

    return {model: np.mean(err) for model, err in errors.items()}


def determine_best_model_combination(model_scores):
    best_combination = []
    min_error = float('inf')

    models = list(model_scores.keys())
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            avg_error = (model_scores[models[i]] + model_scores[models[j]]) / 2
            if avg_error < min_error:
                min_error = avg_error
                best_combination = [models[i], models[j]]

    return best_combination


# # Advanced Data Analysis
# analysis_results = advanced_data_analysis(df)
# print("Advanced Data Analysis Results:", analysis_results)
#
# # Cross Validation of Models
# model_scores = cross_validate_models(df)
# print("Model Cross Validation Scores:", model_scores)
#
# # Best Model Combination
# best_models = determine_best_model_combination(model_scores)
# print("Best Model Combination:", best_models)

def determine_notification_type(VaR, ES, rmse, mae):
    # Встановлення порогових значень
    var_threshold_risk = -0.05
    es_threshold_risk = 0.1
    rmse_threshold_warning = 1000
    mae_threshold_warning = 500

    # Перевірка на критичні ризики
    if VaR <= var_threshold_risk or ES >= es_threshold_risk:
        return "risk"

    # Перевірка на середні ризики
    average_rmse = np.mean(list(rmse.values()))
    average_mae = np.mean(list(mae.values()))
    if average_rmse > rmse_threshold_warning or average_mae > mae_threshold_warning:
        return "warning"

    # Якщо ризики не виявлені
    return "info"


def notification_delivery(forecast, rmse, mae, modelsUsed, models,
                          modelEfficiency, stackingEfficiency, VaR, ES, user_id, code):
    try:
        # Перевірка на наявність прогнозу
        forecast_message = f"Прогноз: {forecast}" if forecast != "No forecast" else "Прогноз відсутній."

        forecast_for_code = f"Прогноз для {code}"

        # Формування повідомлення про використані моделі
        modelsUsed_message = f"Використані моделі: {', '.join(modelsUsed)}."

        # Формування повідомлення про ефективність моделей
        efficiency_message = "Ефективність моделей: " + ', '.join(
            [f"{model}: {efficiency:.2f}%" for model, efficiency in modelEfficiency.items()])

        # Формування повідомлення про метрики
        rmse_message = "RMSE: " + ', '.join([f"{model}: {value:.2f}" for model, value in rmse.items()])
        mae_message = "MAE: " + ', '.join([f"{model}: {value:.2f}" for model, value in mae.items()])

        # Формування повідомлення про стекінгову ефективність
        stackingEfficiency_message = f"Ефективність стекінга: {stackingEfficiency:.2f}%"

        # Формування повідомлення про VaR та ES
        var_message = f"VaR: {VaR:.2f}"
        es_message = f"ES: {ES:.2f}"

        # Збір повного сповіщення
        notification = "\n".join(
            [forecast_for_code, modelsUsed_message, efficiency_message, rmse_message, mae_message,
             stackingEfficiency_message, var_message, es_message])

        notification_type = determine_notification_type(VaR, ES, rmse, mae)

        payload = {
            'userId': user_id,
            'type': notification_type,
            'message': notification,
        }

        post_url = 'http://192.168.1.5:5000/api/userNotifications'

        try:
            post_response = requests.post(post_url, json=payload)
            if post_response.status_code == 200:
                return JsonResponse({
                    "success": True,
                    "message": 'Data successfully saved to database'
                }, status=200)
            else:
                return JsonResponse({
                    "success": False,
                    "message": f"Failed to save data. Status code: {post_response.status_code}"
                }, status=200)
        except requests.exceptions.RequestException as e:
            return JsonResponse({
                "success": False,
                "message": "An error occurred",
                "error": str(e)
            }, status=200)
    except Exception as e:
        return f"Помилка при формуванні сповіщення: {str(e)}"


def get_forecast_data(request, code, user_id):
    url = f'http://192.168.1.5:5000/api/historicalData/{code}'
    response = requests.get(url)

    if response.status_code == 200:
        response_data = response.json()
        data = response_data.get('data')

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

        # Прогнозування майбутніх цін та оцінка моделей
        try:
            model_results = stack_models(df)
            future_prices = model_results['forecast']
            future_returns = calculate_future_returns(future_prices)

            # Розрахунок VaR та ES
            var = calculate_historical_var(future_returns)
            es = calculate_t_dist_es(future_returns)

            # Додавання оригінальних дат та іншої інформації
            historical_data_with_dates = df.assign(date=original_dates).to_dict(orient="records")

        except Exception as e:
            return JsonResponse({"success": False, "response": f"Error during forecasting: {str(e)}"}, status=500)

        response_data = {
            "usedCode": code,
            "historical_data": historical_data_with_dates,
            "forecast": future_prices,
            "VaR": var,
            "ES": es,
            "rmse": model_results['rmse'],
            "mae": model_results['mae'],
            "modelsUsed": model_results['modelsUsed'],
            "models": model_results['models'],
            "modelsEfficiency": model_results['modelsEfficiency'],
            "stackingEfficiency": model_results['stackingEfficiency']
        }

        notification_delivery(
            future_prices,
            model_results['rmse'],
            model_results['mae'],
            model_results['modelsUsed'],
            model_results['models'],
            model_results['modelsEfficiency'],
            model_results['stackingEfficiency'],
            var,
            es,
            user_id,
            code
        )

        date_now_init = datetime.now()
        date_now_plus_2_hours = date_now_init + timedelta(hours=2)
        formatted_date = date_now_plus_2_hours.strftime('%Y-%m-%d %H:%M:%S')
        date_now = formatted_date

        payload = {
            'cryptocurrencyId': 1,
            'cryptoCode': code,
            'forecastDate': date_now,
            'data': response_data
        }

        post_url = 'http://192.168.1.5:5000/api/forecastData'

        try:
            post_response = requests.post(post_url, json=payload)
            if post_response.status_code == 200:
                return JsonResponse({
                    "success": True,
                    "message": 'Data successfully saved to database'
                }, status=200)
            else:
                return JsonResponse({
                    "success": False,
                    "message": f"Failed to save data. Status code: {post_response.status_code}"
                }, status=200)
        except requests.exceptions.RequestException as e:
            return JsonResponse({
                "success": False,
                "message": "An error occurred",
                "error": str(e)
            }, status=200)
    else:
        error_message = response.json().get('error', 'Unknown error')
        return JsonResponse({"success": False, "response": f"Error: {response.status_code}. Message: {error_message}"},
                            status=response.status_code)


# TODO ---------------------------------------------------------------------------------------------------------------

def send_to_database(name, data):
    crypto_code = name.split('_')[2]
    date_now_init = datetime.now()
    date_now_plus_2_hours = date_now_init + timedelta(hours=2)
    formatted_date = date_now_plus_2_hours.strftime('%Y-%m-%d %H:%M:%S')
    date_now = formatted_date
    payload = {
        'cryptocurrencyId': 1,
        'cryptoCode': crypto_code,
        'date': date_now,
        'data': data
    }
    post_url = 'http://192.168.1.5:5000/api/historicalData'

    try:
        post_response = requests.post(post_url, json=payload)
        if post_response.status_code == 200:
            return {"success": True, "message": "Data successfully saved to database", "post_response": post_response}
        else:
            return {"success": False, "message": f"Failed to save data. Status code: {post_response.status_code}",
                    "response": post_response.text}
    except requests.exceptions.RequestException as e:
        return {"success": False, "message": "An error occurred", "error": str(e)}


def get_historical_data(request, name, count, start_time, end_time=None):
    if not end_time:
        end_time = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

    url = f'https://rest.coinapi.io/v1/ohlcv/{name}/history?period_id=1DAY&time_start={start_time}&time_end={end_time}&limit={count}'
    headers = {'X-CoinAPI-Key': 'FF4AACC3-C6FF-47A1-8F4D-5EB8DC574699'}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        db_response = send_to_database(name, data)
        if db_response['success']:
            return JsonResponse({"success": True, "db_response": db_response["message"]}, status=200)
        else:
            return JsonResponse(
                {"success": False, "db_error": db_response["message"], "response": db_response.get("response", "")},
                status=500)
    else:
        return JsonResponse({"success": False, "error": "Request failed"}, status=response.status_code)


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
