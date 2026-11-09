"""Generate clearly synthetic example recordings for a pipeline smoke test.
These are NOT real motor fault measurements and cannot validate fault detection.
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path('data/raw')
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
fs = 1000
n = 8192
for label in ('normal', 'abnormal'):
    for run in range(12):
        t = np.arange(n) / fs
        freq = rng.uniform(45, 65)
        phase = rng.uniform(0, 2 * np.pi)
        signal = 0.2 * np.sin(2 * np.pi * freq * t + phase)
        signal += 0.05 * rng.standard_normal(n)
        if label == 'abnormal':
            signal += 0.15 * np.sin(2 * np.pi * (freq * 2.7) * t + phase)
            impulse_positions = np.arange(200 + run * 3, n, 190)
            signal[impulse_positions] += rng.uniform(0.3, 0.6, len(impulse_positions))
        pd.DataFrame({'time_s': t, 'accel': signal}).to_csv(OUT / f'{label}_run_{run:02d}.csv', index=False)
print('Created 24 synthetic CSV recordings in data/raw/.')
