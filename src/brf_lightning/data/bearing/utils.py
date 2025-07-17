import re
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import hilbert

def _parse_speed(name: str) -> int:
    """Extract speed in Hz from a file name, e.g. 'fAII20.csv' → 20."""
    return int("".join(ch for ch in name.stem if ch.isdigit()))

def _bearing_id_from_path(fp: str | Path) -> str:
    """
    Extract a *bearing identifier* from the filename or folder.
    Adapt the pattern to your naming scheme – the key point is:
    all CSVs originating from the **same physical bearing** must
    return the same string here.
    """
    fp = Path(fp)
    # Example: ".../bearing03_speed20_inner.csv"  -->  "bearing03"
    return fp.stem.split('_')[0]

def _rpm_from_path(fp: str | Path) -> int:
    """Extract the first integer from a filename as nominal RPM."""
    return int("".join(c for c in Path(fp).stem if c.isdigit()))

def _class_id_from_path(fp: str | Path) -> int:
    """
    Map filename → integer label   (healthy=0, inner-race=1, outer-race=2, …).
    Re-use your existing helper if you prefer.
    """
    name = Path(fp).stem.lower()
    if   "n"   in name: return 0
    elif "fa"     in name: return 1
    elif "fi"     in name: return 2
    else: raise ValueError(f"Cannot parse class from {fp}")

def load_vibration_data(path: str | Path) -> tuple[pd.DataFrame, float]:
    """
    Load vibration data from a CSV file whose first line encodes
    the sampling rate, then drop empty columns and rename the data columns.

    The file is expected to look like:
        fs:8192;
        sensor1;sensor2;trig;
        0.12;0.34;1;
        ...

    This function will:
      1. Read the first line and parse `fs:<number>;`
      2. Load the rest of the file into a DataFrame
      3. Drop the trailing unnamed column if it is entirely NaN
      4. Rename columns to ['sensor_bearing', 'sensor_engine', 'trig']
      5. Cast the trigger column to int

    Parameters
    ----------
    fp : str
        Path to the CSV file.

    Returns
    -------
    df : pd.DataFrame
        Cleaned vibration data with columns:
        ['sensor_bearing', 'sensor_engine', 'trig'].
    fs : float
        Sampling rate, in Hz, extracted from the header.
    """
    if isinstance(path, str):
        path = Path(path)
    # 1. Extract sampling rate
    header = path.open("r").readline().strip()
    m = re.match(r"^fs:(?P<rate>\d+(?:\.\d+)?);", header)
    if not m:
        raise ValueError(f"Could not parse sampling rate from header: {header!r}")
    fs = float(m.group("rate"))

    # 2. Load the data skipping the header line
    df = pd.read_csv(path, delimiter=';', skiprows=1)

    # 3. Drop the last column if unnamed and entirely NaN
    if df.columns[-1].startswith("Unnamed") and df.iloc[:, -1].isna().all():
        df = df.iloc[:, :-1]

    # 4. Rename columns
    df.columns = ['sensor_bearing', 'sensor_engine', 'trig']

    # 5. Ensure triggers are integers
    df['trig'] = df['trig'].astype(int)

    return df, fs


# ---------- signal utilities --------------------------------------------------
def _highpass(sig: np.ndarray, fs: float, fc: float = 2.0, order: int = 4) -> np.ndarray:
    """Butterworth high-pass using SciPy.  Falls back to identity if SciPy is absent."""
    try:
        from scipy.signal import butter, filtfilt
        b, a = butter(order, fc / (0.5 * fs), btype="high", analog=False)
        return filtfilt(b, a, sig, padlen=3 * order)
    except ImportError:
        return sig  # graceful degradation

def fdtw_safe(
    x: np.ndarray,
    fs: float,
    *,
    spr: int = 360,
    max_len_sec: float = 30.0,
) -> tuple[np.ndarray, float]:
    """Tacholess order-domain resample with a RAM cap."""
    if x.size > max_len_sec * fs:
        x = x[: int(max_len_sec * fs)]

    phase = np.unwrap(np.angle(hilbert(x)))
    grid  = np.arange(phase[0], phase[-1], 2 * np.pi / spr)
    x_ord = np.interp(grid, phase, x).astype(np.float32)
    fs_ord = float(spr)               # samples per revolution
    return x_ord, fs_ord
