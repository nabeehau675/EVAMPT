"""Classify the first 1024 acceleration readings in a CSV recording."""
import json
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

if len(sys.argv) != 2:
    raise SystemExit('Usage: python predict.py data/raw/normal_run_00.csv')
settings = json.load(open('models/normalization.json', encoding='utf-8'))
values = pd.read_csv(sys.argv[1])['accel'].to_numpy(dtype=np.float32)
n = settings['window']
if len(values) < n:
    raise SystemExit(f'Need at least {n} acceleration samples')
x = ((values[:n] - settings['mean']) / settings['std'])[None, :, None]
model = tf.keras.models.load_model('models/hackedge_vibration.keras')
probability = float(model.predict(x, verbose=0)[0, 0])
print(f'Estimated abnormal probability: {probability:.3f}')
print('Prediction:', 'abnormal' if probability >= 0.5 else 'normal')
