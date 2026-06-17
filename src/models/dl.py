"""Deep learning anomaly detector: LSTM Autoencoder."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .base import AnomalyDetector


def _make_windows(X: np.ndarray, window: int) -> np.ndarray:
    """Return rolling windows of shape (N - window + 1, window, n_features)."""
    n = len(X)
    if n < window:
        # pad with edge values
        pad = np.tile(X[0], (window - n, 1))
        X = np.vstack([pad, X])
        n = len(X)
    return np.stack([X[i : i + window] for i in range(n - window + 1)])


class _LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32, window: int = 20):
        super().__init__()
        self.window = window
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        # repeat hidden across window steps
        rep = h.squeeze(0).unsqueeze(1).repeat(1, self.window, 1)
        dec, _ = self.decoder(rep)
        return self.out(dec)


class LSTMAutoencoderDetector(AnomalyDetector):
    name = "LSTM-Autoencoder"

    def __init__(
        self,
        contamination: float = 0.05,
        window: int = 20,
        hidden: int = 32,
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 1e-3,
        **kwargs,
    ):
        super().__init__(contamination=contamination, **kwargs)
        self.window = window
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def fit_score(self, X: pd.DataFrame) -> pd.Series:
        torch.manual_seed(42)
        Xv = X.values.astype(np.float32)
        window = max(5, min(self.window, max(5, len(Xv) // 4)))
        windows = _make_windows(Xv, window).astype(np.float32)
        tensor = torch.from_numpy(windows)

        device = torch.device("cpu")
        model = _LSTMAutoencoder(
            n_features=Xv.shape[1], hidden=self.hidden, window=window
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        model.train()
        n = len(tensor)
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                batch = tensor[idx].to(device)
                recon = model(batch)
                loss = loss_fn(recon, batch)
                opt.zero_grad()
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            recon = model(tensor.to(device)).cpu().numpy()
        # per-window MSE across last timestep features
        last_true = windows[:, -1, :]
        last_recon = recon[:, -1, :]
        per_point_err = np.mean((last_true - last_recon) ** 2, axis=1)

        # align scores back to original index: pad first (window-1) with edge value
        scores = np.concatenate(
            [np.full(window - 1, per_point_err[0]), per_point_err]
        )
        scores = scores[: len(X)]
        return pd.Series(scores, index=X.index, name="score")
