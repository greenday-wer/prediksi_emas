import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from math import sqrt
from datetime import timedelta
import streamlit as st

st.set_page_config(page_title="Prediksi Harga Emas", page_icon="💰", layout="wide")

st.title("📈 Dashboard Prediksi Harga Emas")
future_days = st.slider("Berapa Hari ke Depan yang Ingin Diprediksi?", 5, 30, 10)

@st.cache_data
def fetch_gold_data():
    df = pd.read_csv("DataEmas.csv")
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], format="%d/%m/%Y")

    def parse_price(price_str):
        return float(price_str.replace('.', '').replace(',', '.'))

    df['Harga'] = df['Terakhir'].apply(parse_price)

    df = df.sort_values('Tanggal')
    df = df.reset_index(drop=True)

    return df

df = fetch_gold_data()

harga_terbaru = df.loc[df['Tanggal'] == df['Tanggal'].max(), 'Harga'].values[0]
tanggal_awal = df['Tanggal'].min().strftime('%d-%m-%Y')
tanggal_akhir = df['Tanggal'].max().strftime('%d-%m-%Y')

st.subheader(f"Harga Emas Terakhir: **US ${harga_terbaru:,.2f}**")
st.caption(f"Data dari tanggal {tanggal_awal} hingga {tanggal_akhir}")

# XGBoost Model
sequence_length = 30
df['Hari'] = np.arange(len(df))

X_xgb = []
y_xgb = []
for i in range(sequence_length, len(df)):
    X_xgb.append(df['Harga'].values[i-sequence_length:i])
    y_xgb.append(df['Harga'].values[i])

X_xgb = np.array(X_xgb)
y_xgb = np.array(y_xgb)

model_xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100)
model_xgb.fit(X_xgb, y_xgb)

last_seq_xgb = df['Harga'].values[-sequence_length:]
pred_xgb = []
for _ in range(future_days):
    input_seq = np.array(last_seq_xgb[-sequence_length:]).reshape(1, -1)
    pred = model_xgb.predict(input_seq)[0]
    pred_xgb.append(pred)
    last_seq_xgb = np.append(last_seq_xgb, pred)

# LSTM Model
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Harga']])

X_lstm, y_lstm = [], []
for i in range(sequence_length, len(scaled_data)):
    X_lstm.append(scaled_data[i-sequence_length:i])
    y_lstm.append(scaled_data[i])

X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model_lstm = Sequential([
    Input(shape=(X_lstm.shape[1], 1)),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X_lstm, y_lstm, epochs=50, batch_size=16, verbose=0, validation_split=0.1, callbacks=[early_stopping])

last_seq = scaled_data[-sequence_length:]
preds_scaled = []

for _ in range(future_days):
    input_seq = last_seq[-sequence_length:].reshape(1, sequence_length, 1)
    pred = model_lstm.predict(input_seq, verbose=0)
    preds_scaled.append(pred[0][0])
    last_seq = np.append(last_seq, pred, axis=0)

pred_lstm = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1))

future_dates = pd.date_range(df['Tanggal'].iloc[-1] + timedelta(days=1), periods=future_days)

tooltip_cols = ['Terakhir', 'Pembukaan', 'Tertinggi', 'Terendah', 'Vol.', 'Perubahan%']
for col in tooltip_cols:
    df[col] = df[col].astype(str)

customdata = df[tooltip_cols].values

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['Tanggal'],
    y=df['Harga'],
    name='Harga Historis',
    mode='lines',
    line=dict(color='blue'),
    customdata=customdata,
    hovertemplate=
        'Tanggal: %{x|%d-%m-%Y}<br>' +
        'Terakhir: %{customdata[0]}<br>' +
        'Pembukaan: %{customdata[1]}<br>' +
        'Tertinggi: %{customdata[2]}<br>' +
        'Terendah: %{customdata[3]}<br>' +
        'Vol.: %{customdata[4]}<br>' +
        'Perubahan%%: %{customdata[5]}<extra></extra>'
))

fig.add_trace(go.Scatter(
    x=future_dates,
    y=pred_xgb,
    name='XGBoost',
    line=dict(dash='dot')
))

fig.add_trace(go.Scatter(
    x=future_dates,
    y=pred_lstm.flatten(),
    name='LSTM',
    line=dict(dash='dash')
))

fig.update_layout(
    title='Prediksi Harga Emas',
    xaxis_title='Tanggal',
    yaxis_title='Harga (USD)',
    template='plotly_white',
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

if len(df) > future_days + sequence_length:
    y_true = df['Harga'].iloc[-future_days:].values
    rmse_xgb = sqrt(mean_squared_error(y_true, pred_xgb[:len(y_true)]))
    rmse_lstm = sqrt(mean_squared_error(y_true, pred_lstm[:len(y_true)]))
    st.metric("RMSE XGBoost", f"{rmse_xgb:.2f}")
    st.metric("RMSE LSTM", f"{rmse_lstm:.2f}")

df_result = pd.DataFrame({
    "Tanggal": future_dates.strftime("%Y-%m-%d"),
    "Prediksi XGBoost (USD)": pred_xgb,
    "Prediksi LSTM (USD)": pred_lstm.flatten()
})

st.subheader("📊 Tabel Prediksi")
st.dataframe(df_result.set_index("Tanggal"), use_container_width=True)

st.markdown("""
    <style>
    #MainMenu, header, footer {visibility: hidden;}
    .block-container {padding-top: 2rem;}
    </style>
""", unsafe_allow_html=True)