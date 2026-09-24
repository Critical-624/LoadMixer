"""CSV input and chronological forecasting windows."""
import hashlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def read_csv(path, target, date_column="date"):
    frame = pd.read_csv(path)
    if target == date_column or target not in frame or date_column not in frame:
        raise ValueError(f"CSV must contain distinct '{date_column}' and '{target}' columns.")
    dates = pd.to_datetime(frame[date_column], errors="raise")
    values = pd.to_numeric(frame[target], errors="raise").to_numpy(dtype=np.float64)
    if dates.isna().any() or not np.isfinite(values).all():
        raise ValueError("Timestamps and target values must not contain missing or non-finite values.")
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError("Rows must be in increasing timestamp order without duplicates.")
    steps = dates.diff().dropna()
    if len(steps) and not steps.eq(steps.iloc[0]).all():
        raise ValueError("Timestamps must be equally spaced. Aggregate or resample your CSV first.")
    marks = np.column_stack([
        dates.dt.minute / 59 - 0.5,
        dates.dt.hour / 23 - 0.5,
        dates.dt.dayofweek / 6 - 0.5,
        (dates.dt.day - 1) / 30 - 0.5,
        (dates.dt.dayofyear - 1) / 365 - 0.5,
    ]).astype(np.float32)
    return dates, values, marks


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Windows(Dataset):
    """Targets stay inside the split; history may precede its boundary."""

    def __init__(self, values, marks, start, end, history, horizon, mean, scale):
        self.first_target = max(start, history)
        self.count = end - horizon - self.first_target + 1
        if self.count <= 0:
            raise ValueError("Not enough rows for a forecasting window in this split. Supply a longer CSV.")
        self.history, self.horizon = history, horizon
        normalized = ((values - mean) / scale).astype(np.float32)
        if not np.isfinite(normalized).all():
            raise ValueError("Values exceed the supported floating-point range after scaling.")
        self.values = torch.from_numpy(normalized[:, None])
        self.marks = torch.from_numpy(marks)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        start = self.first_target + index
        return (
            self.values[start - self.history:start],
            self.marks[start - self.history:start],
            self.values[start:start + self.horizon],
        )
