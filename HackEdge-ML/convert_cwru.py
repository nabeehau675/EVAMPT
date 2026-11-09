
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import resample_poly
import pandas as pd

source = Path("data/external/cwru")
destination = Path("data/real")
destination.mkdir(parents=True, exist_ok=True)

healthy_files = [97, 98, 99, 100]
faulty_files = [105, 106, 107, 108]

# Healthy recordings: 48 kHz -> 12 kHz
for number in healthy_files:
    data = loadmat(source / f"{number}.mat")
    key = f"X{number:03d}_DE_time"
    vibration = data[key].ravel()

    # Anti-alias filtering and downsampling by 4
    vibration = resample_poly(vibration, up=1, down=4)

    output = destination / f"normal_{number}.csv"
    pd.DataFrame({"accel": vibration}).to_csv(output, index=False)

    print(f"Healthy: {output.name}, {len(vibration)} samples")

# Faulty recordings: already 12 kHz
for number in faulty_files:
    data = loadmat(source / f"{number}.mat")
    key = f"X{number:03d}_DE_time"
    vibration = data[key].ravel()

    output = destination / f"abnormal_{number}.csv"
    pd.DataFrame({"accel": vibration}).to_csv(output, index=False)

    print(f"Faulty: {output.name}, {len(vibration)} samples")

print("All recordings converted to 12 kHz!")
