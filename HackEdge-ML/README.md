# HackEdge vibration 1D-CNN starter

This project trains a **binary, single-axis vibration classifier** on a laptop. It does **not** require the TI SK-AM62B EVM to train.

## Windows / VS Code quick start

1. Install Python 3.11 (64-bit) and VS Code with the Python extension.
2. Open this folder in VS Code and open Terminal > New Terminal.
3. Run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe generate_demo_data.py
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe predict.py data/raw/normal_run_00.csv
```

If `py -3.11` is not available, install Python 3.11 or use a compatible Python interpreter. No need to activate the environment with PowerShell.

## Replacing demo data with real vibration recordings

Delete/move the synthetic demo CSVs out of `data/raw/`, then put each independent recording in its own CSV. Use filenames like `normal_run_01.csv` or `abnormal_run_01.csv` and a numeric `accel` column. Optional `time_s` column is ignored. Aim for at least 10 independent recordings per class; more is better. All recordings must use a **consistent sample rate**, sensor axis, units, and sensor mounting setup. The sample rate should be documented (for example, 1000 Hz) and chosen based on the motor's expected fault frequencies. Each recording must contain at least 1024 samples.

The code splits **recordings** into training, validation and test sets before making 1024-sample windows to reduce data leakage. Normalization is learned from training data only. For rigorous performance estimates, collect data across different runs, loads and motor speeds, and test on conditions not seen during training. If the same motor/rig appears in all splits, the measured accuracy may still overestimate performance on other hardware.

## What the demo means

`generate_demo_data.py` creates artificial normal and abnormal waveforms solely to verify that code runs end to end. A high test score on synthetic data is **not evidence** of real motor fault detection. Avoid interpreting demo predictions as predictive-maintenance capability.

## Files

- `generate_demo_data.py`: synthetic recordings for software smoke testing
- `preprocess.py`: loads recordings, splits by run, windows and normalizes
- `train.py`: trains a compact 3-convolution-layer 1D-CNN and evaluates it
- `predict.py`: classifies a 1024-sample segment
- `models/`: saved model and normalization settings

## Deployment later

The `.keras` model is for training/reuse on your laptop. Later, verify an inference runtime supported by the exact TI Linux image and architecture, convert/quantize if appropriate, and test accuracy and latency on the EVM. Conversion does not guarantee runtime compatibility.
