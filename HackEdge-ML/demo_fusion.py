
"""Test EVAMPT sensor fusion using the existing vibration CNN."""

import json
import numpy as np

from fusion import Baseline, SensorWindow, VibrationModel, analyze

# Step 1: Load the trained vibration CNN
print("Loading vibration CNN...")

vibration_model = VibrationModel(
    model_path="models/hackedge_vibration_cwru.tflite",
    normalization_path="models/normalization_cwru.json"
)


print("CNN loaded successfully!")

# Step 2: Generate demonstration sensor measurements
rng = np.random.default_rng(42)

window = SensorWindow(
    vibration=rng.normal(0, 0.1, size=12000).astype(np.float32),
    voltage=np.array([4.9, 4.8, 4.9]),
    current=np.array([0.5, 0.55, 0.52]),
    temperature_c=43.0,
    rpm=1600.0,
    duration_s=1.0
)

# Step 3: Run the actual CNN and sensor fusion
print("Running CNN and sensor fusion...")

result = analyze(
    window,
    Baseline(),
    vibration_model=vibration_model
)

# Step 4: Display results
print(json.dumps(result, indent=2))
