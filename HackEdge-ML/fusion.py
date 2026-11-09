"""EVAMPT hybrid sensor fusion prototype (not a validated safety monitor)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np


@dataclass
class SensorWindow:
    # All waveforms correspond to the same time window; vibration sampled at 12 kHz.
    vibration: np.ndarray  # (N,) single-axis, or (N, 3) for XYZ
    voltage: np.ndarray  # volts, samples across window
    current: np.ndarray  # amps, samples across window
    temperature_c: float
    rpm: float
    duration_s: float


@dataclass
class Baseline:
    temperature_c: float = 30.0
    rpm: float = 1800.0
    current_a: float = 0.35
    voltage_v: float = 5.0
    max_temp_c: float = 60.0
    max_current_a: float = 0.75
    min_rpm_fraction: float = 0.75
    min_voltage_fraction: float = 0.85


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _require_array(value, name: str) -> np.ndarray:
    x = np.asarray(value, dtype=np.float32).reshape(-1)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError(f'{name} must contain finite numeric samples')
    return x


def vibration_features(samples, sample_rate_hz: float = 12000.0) -> dict:
    x = _require_array(samples, 'vibration')
    centered = x - x.mean()
    rms = float(np.sqrt(np.mean(centered**2)))
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(len(centered))))
    frequencies = np.fft.rfftfreq(len(centered), 1 / sample_rate_hz)
    dominant_hz = float(frequencies[np.argmax(spectrum[1:]) + 1]) if len(spectrum) > 1 else 0.0
    return {'vibration_rms': rms, 'vibration_peak': float(np.max(np.abs(centered))),
            'dominant_frequency_hz': dominant_hz}


class VibrationModel:
    """Loads the existing single-axis (1, 1024, 1) vibration TFLite classifier."""
    def __init__(self, model_path='models/hackedge_vibration.tflite',
                 normalization_path='models/normalization.json'):
        
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        with open(normalization_path, encoding='utf-8') as file:
            params = json.load(file)
        # Adjust these keys to match the normalization.json created by your train.py.
        self.mean = float(params['mean'])
        self.std = float(params['std'])
        if self.std <= 0:
            raise ValueError('Normalization standard deviation must be positive')
        shape = tuple(int(n) for n in self.input_detail['shape'])
        if shape != (1, 1024, 1):
            raise ValueError(f'Expected CNN input (1, 1024, 1), found {shape}')

    def predict(self, samples) -> float:
        x = _require_array(samples, 'vibration')
        if x.size < 1024:
            raise ValueError('Vibration window requires at least 1024 samples')
        # Average non-overlapping 1024-sample predictions across the window.
        predictions = []
        for start in range(0, x.size - 1024 + 1, 1024):
            chunk = ((x[start:start + 1024] - self.mean) / self.std)
            chunk = chunk.reshape(1, 1024, 1).astype(np.float32)
            self.interpreter.set_tensor(self.input_detail['index'], chunk)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_detail['index'])
            predictions.append(float(output.reshape(-1)[0]))
        return _clip01(float(np.mean(predictions)))


def analyze(window: SensorWindow, baseline: Baseline, vibration_model=None,
            vibration_anomaly_score=None) -> dict:
    """Use a CNN score or an explicitly supplied mock score; no implicit fake prediction."""
    vib = np.asarray(window.vibration, dtype=np.float32)
    if vib.ndim == 2 and vib.shape[1] == 3:
        # Existing CWRU CNN is single-axis: use X only for prototype.
        x = vib[:, 0]
    elif vib.ndim == 1:
        x = vib
    else:
        raise ValueError('vibration must have shape (N,) or (N, 3)')
    x = _require_array(x, 'vibration')
    voltage = _require_array(window.voltage, 'voltage')
    current = _require_array(window.current, 'current')
    if not all(np.isfinite(v) for v in [window.temperature_c, window.rpm, window.duration_s]):
        raise ValueError('temperature, RPM and duration must be finite')
    if window.duration_s <= 0 or window.rpm < 0:
        raise ValueError('duration must be positive and RPM nonnegative')
    if vibration_model is not None:
        vibration_score = vibration_model.predict(x)
    elif vibration_anomaly_score is not None:
        vibration_score = _clip01(float(vibration_anomaly_score))
    else:
        raise ValueError('Provide a vibration_model or explicit mock vibration_anomaly_score')

    v_mean = float(np.mean(voltage))
    i_mean = float(np.mean(current))
    electrical_score = max(
        _clip01((i_mean - baseline.current_a) / max(baseline.max_current_a - baseline.current_a, 1e-6)),
        _clip01((baseline.voltage_v - v_mean) / max(baseline.voltage_v * (1 - baseline.min_voltage_fraction), 1e-6))
    )
    temp_score = _clip01((window.temperature_c - baseline.temperature_c) /
                         max(baseline.max_temp_c - baseline.temperature_c, 1e-6))
    rpm_score = _clip01((baseline.rpm - window.rpm) /
                        max(baseline.rpm * (1 - baseline.min_rpm_fraction), 1e-6))
    scores = {'vibration': vibration_score, 'electrical': electrical_score,
              'temperature': temp_score, 'rpm': rpm_score}
    weights = {'vibration': 0.40, 'electrical': 0.30, 'temperature': 0.15, 'rpm': 0.15}
    health_anomaly_score = sum(weights[k] * scores[k] for k in scores)
    status = 'FAULT_INDICATOR' if health_anomaly_score >= 0.65 else (
        'WARNING' if health_anomaly_score >= 0.40 else 'NORMAL_INDICATOR')
    explanations = [f'{k} anomaly score {score:.2f}' for k, score in scores.items() if score >= 0.5]
    features = vibration_features(x)
    features.update({'voltage_mean_v': v_mean, 'current_mean_a': i_mean,
                     'electrical_power_estimate_w': v_mean * i_mean,
                     'temperature_c': float(window.temperature_c), 'rpm': float(window.rpm)})
    return {'status': status, 'anomaly_score': round(health_anomaly_score, 4),
            'module_scores': {k: round(v, 4) for k, v in scores.items()},
            'features': features, 'reasons': explanations,
            'note': 'Prototype thresholds/weights; not validated for fault diagnosis.'}
