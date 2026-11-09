"""Load one-column accelerometer CSV recordings; split by recording before windowing."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

WINDOW = 1024
STRIDE = 512
LABELS = {'normal': 0, 'abnormal': 1}

def load_recordings(folder='data/real'):
    recordings = []
    for path in sorted(Path(folder).glob('*.csv')):
        label = next((value for name, value in LABELS.items() if path.stem.startswith(name + '_')), None)
        if label is None:
            continue
        frame = pd.read_csv(path)
        if 'accel' not in frame.columns:
            raise ValueError(f'{path}: expected an accel column')
        samples = frame['accel'].to_numpy(dtype=np.float32)
        if not np.isfinite(samples).all():
            raise ValueError(f'{path}: NaN or infinite values')
        if len(samples) < WINDOW:
            raise ValueError(f'{path}: fewer than {WINDOW} samples')
        recordings.append((path.name, samples, label))
    if not recordings:
        raise FileNotFoundError('No labeled CSVs found. Expected normal_*.csv and abnormal_*.csv in data/raw/')
    return recordings

def make_windows(recordings, mean, std):
    x, y = [], []
    for _, samples, label in recordings:
        for start in range(0, len(samples) - WINDOW + 1, STRIDE):
            window = (samples[start:start + WINDOW] - mean) / std
            x.append(window[:, None])
            y.append(label)
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)

def prepare_data():
    recordings = load_recordings()
    normal = sorted((r for r in recordings if r[2] == 0), key=lambda r: r[0])
    abnormal = sorted((r for r in recordings if r[2] == 1), key=lambda r: r[0])

    if len(normal) != 4 or len(abnormal) != 4:
        raise ValueError("Expected exactly 4 normal and 4 abnormal recordings")

    train = normal[:2] + abnormal[:2]
    val = normal[2:3] + abnormal[2:3]
    test = normal[3:] + abnormal[3:]

    print("Training recordings:", [r[0] for r in train])
    print("Validation recordings:", [r[0] for r in val])
    print("Testing recordings:", [r[0] for r in test])
    # Compute normalization ONLY from training recordings to avoid leakage.
    all_train = np.concatenate([r[1] for r in train])
    mean = float(all_train.mean())
    std = float(all_train.std())
    if std < 1e-8:
        raise ValueError('Training signals have almost zero variation')
    splits = [make_windows(group, mean, std) for group in (train, val, test)]
    print('Recording counts (train, val, test):', len(train), len(val), len(test))
    print('Window counts:', [len(s[1]) for s in splits])
    return splits, mean, std
