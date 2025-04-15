import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Function to create sequences
def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Streamlit UI
st.title("📈 LSTM Stock Price Predictor")
st.markdown("Enter stock details to forecast stock prices using an LSTM model.")

# Sidebar inputs
stock = st.text_input("Stock Ticker", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2023-10-01"))

if st.button("Predict"):
    df = yf.download(stock, start=start_date, end=end_date)

    if df.empty:
        st.error("No data found. Check the ticker or date range.")
    else:
        data = df[['Close']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)

        time_steps = 60
        X, y = create_sequences(data_scaled, time_steps)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        y_pred = model.predict(X_test)
        y_pred_rescaled = scaler.inverse_transform(y_pred)
        y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Metrics
        r2 = r2_score(y_test_rescaled, y_pred_rescaled)
        mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
        mape = np.mean(np.abs((y_test_rescaled - y_pred_rescaled) / y_test_rescaled)) * 100
        accuracy = 100 - mape

        st.subheader("📊 Model Metrics")
        st.write(f"**R² Score:** {r2:.4f}")
        st.write(f"**MAE:** {mae:.4f}")
        st.write(f"**MAPE:** {mape:.2f}%")
        st.write(f"**Accuracy:** {accuracy:.2f}%")

        # Plot actual vs predicted
        st.subheader("📉 Actual vs Predicted Prices")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(y_test_rescaled, label="Actual")
        ax.plot(y_pred_rescaled, label="Predicted")
        ax.legend()
        st.pyplot(fig)

        # Future prediction
        st.subheader("📆 Future 30-Day Forecast")
        future_input = data_scaled[-time_steps:].reshape(1, time_steps, 1)
        future_predictions = []
        for _ in range(30):
            next_pred = model.predict(future_input, verbose=0)[0][0]
            future_predictions.append(next_pred)
            future_input = np.append(future_input[:, 1:, :], [[[next_pred]]], axis=1)

        future_predictions_rescaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        future_dates = pd.date_range(start=df.index[-1], periods=31, freq='B')[1:]

        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.plot(df.index[-100:], df['Close'].values[-100:], label="Past Prices")
        ax2.plot(future_dates, future_predictions_rescaled, linestyle='dashed', label="Future Forecast")
        ax2.legend()
        st.pyplot(fig2)
