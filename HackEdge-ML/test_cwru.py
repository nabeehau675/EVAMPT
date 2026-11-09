
"""Test the CWRU-trained TFLite model on held-out vibration recordings."""

from pathlib import Path

import numpy as np
import pandas as pd

from fusion import VibrationModel

model = VibrationModel(
    model_path="models/hackedge_vibration_cwru.tflite",
    normalization_path="models/normalization_cwru.json"
)

# These two recordings were reserved for testing.
test_files = [
    ("data/real/normal_99.csv", "NORMAL"),
    ("data/real/abnormal_108.csv", "ABNORMAL"),
]

print("\nEVAMPT - REAL CWRU VIBRATION TEST")
print("=" * 45)

for filename, expected in test_files:
    if not Path(filename).exists():
        raise FileNotFoundError(filename)

    samples = pd.read_csv(filename)["accel"].to_numpy(dtype=np.float32)

    anomaly_score = model.predict(samples)
    predicted = "ABNORMAL" if anomaly_score >= 0.5 else "NORMAL"

    print(f"\nRecording: {Path(filename).name}")
    print(f"Expected: {expected}")
    print(f"Predicted: {predicted}")
    print(f"Anomaly score: {anomaly_score:.4f}")
    print(f"Result: {'PASS' if predicted == expected else 'FAIL'}")

print("\nTest complete!")
