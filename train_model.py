import sys
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense

if len(sys.argv) < 2:
    print("Error: Missing dataset argument.")
    print("Usage: python train_model.py <dataset.csv>")
    sys.exit(1)

file_path = sys.argv[1]

if not os.path.exists(file_path):
    print(f"Error: The file '{file_path}' was not found.")
    sys.exit(1)

print(f"Initializing Edge AI Pipeline for: {file_path}")

df = pd.read_csv(file_path)

drop_keywords = ['time', 'timestamp', 'date', 'label']
cols_to_drop = [col for col in df.columns if col.lower() in drop_keywords]
data = df.drop(columns=cols_to_drop)

data = data.select_dtypes(include=[np.number])

print(f"Detected {len(data.columns)} raw sensors. Engineering variance features...")

ROLLING_WINDOW = 5
original_cols = list(data.columns)

for col in original_cols:
    data[f'{col}_variance'] = data[col].rolling(window=ROLLING_WINDOW).var()

data = data.fillna(0)

print(f"Feature engineering complete. Total inputs per timestep: {len(data.columns)}")

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

def create_sequence(dataset, time_steps):
    sequence = []
    for i in range(len(dataset) - time_steps):
        seq = dataset[i:(i + time_steps)]
        sequence.append(seq)
    return np.array(sequence)

TIME_STEPS = 10
X_train = create_sequence(scaled_data, TIME_STEPS)
NUM_FEATURES = X_train.shape[2] 

print("Building Lightweight Edge AI Architecture (~3.3k parameters)...")
model = Sequential([
    LSTM(16, activation='relu', input_shape=(TIME_STEPS, NUM_FEATURES), return_sequences=False),
    RepeatVector(TIME_STEPS),
    LSTM(16, activation='relu', return_sequences=True),
    TimeDistributed(Dense(NUM_FEATURES))
])

model.compile(optimizer='adam', loss='mse')

print("Training model...")
model.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

model.save("anomaly_detection_model.keras")
joblib.dump(scaler, "scaler.pkl")

config = {
    'time_steps': TIME_STEPS,
    'num_features': NUM_FEATURES,
    'feature_names': list(data.columns)
}
joblib.dump(config, "model_config.pkl")

print("Pipeline Complete! Model, Scaler, and Config saved successfully.")