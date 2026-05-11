
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "season", "yr", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "temp", "atemp", "hum", "windspeed",
]


def make_seq(x: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(len(x) - seq_len + 1):
        xs.append(x[i : i + seq_len])
        ys.append(y[i + seq_len - 1])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


class SimpleRNNRegressor:
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        learning_rate: float = 1e-3,
        max_epochs: int = 25,
        batch_size: int = 128,
        patience: int = 7,
        grad_clip: float = 10.0,
        random_state: int = 43,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.grad_clip = grad_clip

        self.rng = np.random.default_rng(random_state)

        self.W_xh: np.ndarray | None = None
        self.W_hh: np.ndarray | None = None
        self.b_h: np.ndarray | None = None
        self.W_hy: np.ndarray | None = None
        self.b_y: np.ndarray | None = None

        self._init_params()

    def _init_params(self) -> None:
        scale_in = np.sqrt(1.0 / max(self.input_size, 1))
        scale_h = np.sqrt(1.0 / max(self.hidden_size, 1))
        self.W_xh = self.rng.standard_normal((self.input_size, self.hidden_size), dtype=np.float32) * scale_in
        self.W_hh = self.rng.standard_normal((self.hidden_size, self.hidden_size), dtype=np.float32) * scale_h
        self.b_h = np.zeros((1, self.hidden_size), dtype=np.float32)
        self.W_hy = self.rng.standard_normal((self.hidden_size, 1), dtype=np.float32) * scale_h
        self.b_y = np.zeros((1, 1), dtype=np.float32)
        self._adam_t = 0
        self._adam_m = {name: np.zeros_like(getattr(self, name)) for name in self._param_names()}
        self._adam_v = {name: np.zeros_like(getattr(self, name)) for name in self._param_names()}

    @staticmethod
    def _adam_hyperparams() -> tuple[float, float, float]:
        return 0.9, 0.999, 1e-8

    @staticmethod
    def _param_names() -> tuple[str, ...]:
        return "W_xh", "W_hh", "b_h", "W_hy", "b_y"

    def _snapshot_state(self) -> dict[str, object]:
        return {
            "params": {name: getattr(self, name).copy() for name in self._param_names()},
            "adam_m": {name: value.copy() for name, value in self._adam_m.items()},
            "adam_v": {name: value.copy() for name, value in self._adam_v.items()},
            "adam_t": self._adam_t,
        }

    def _restore_state(self, state: dict[str, object]) -> None:
        params = state["params"]
        adam_m = state["adam_m"]
        adam_v = state["adam_v"]
        self._adam_t = int(state["adam_t"])
        for name, value in params.items():
            setattr(self, name, value.copy())
        self._adam_m = {name: value.copy() for name, value in adam_m.items()}
        self._adam_v = {name: value.copy() for name, value in adam_v.items()}

    def _forward(self, x_batch: np.ndarray) -> tuple[np.ndarray, dict[str, list[np.ndarray] | np.ndarray]]:
        batch_size, seq_len, _ = x_batch.shape
        h_prev = np.zeros((batch_size, self.hidden_size), dtype=np.float32)

        h_list: list[np.ndarray] = []
        h_prev_list: list[np.ndarray] = []
        x_list: list[np.ndarray] = []

        for t in range(seq_len):
            x_t = x_batch[:, t, :]
            h_prev_list.append(h_prev)
            h_t = np.tanh(np.dot(x_t, self.W_xh) + np.dot(h_prev, self.W_hh) + self.b_h)
            x_list.append(x_t)
            h_list.append(h_t)
            h_prev = h_t

        y_pred = np.dot(h_list[-1], self.W_hy) + self.b_y
        cache = {"x_list": x_list, "h_list": h_list, "h_prev_list": h_prev_list, "y_pred": y_pred}
        return y_pred, cache

    def _clip_grads(self, grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        total_norm = 0.0
        for g in grads.values():
            total_norm += float(np.sum(g * g))
        total_norm = np.sqrt(total_norm)

        if total_norm > self.grad_clip and total_norm > 0:
            scale = self.grad_clip / total_norm
            for k in grads:
                grads[k] *= scale
        return grads

    def _backward(self, y_true: np.ndarray, cache: dict[str, list[np.ndarray] | np.ndarray]) -> dict[str, np.ndarray]:
        x_list = cache["x_list"]
        h_list = cache["h_list"]
        h_prev_list = cache["h_prev_list"]
        y_pred = cache["y_pred"]

        n = y_true.shape[0]
        y_true_2d = y_true.reshape(-1, 1)

        d_y = (2.0 * (y_pred - y_true_2d)) / max(n, 1)

        d_W_hy = np.dot(h_list[-1].T, d_y)
        d_b_y = np.sum(d_y, axis=0, keepdims=True)

        d_W_xh = np.zeros_like(self.W_xh)
        d_W_hh = np.zeros_like(self.W_hh)
        d_b_h = np.zeros_like(self.b_h)

        d_h = np.dot(d_y, self.W_hy.T)

        for t in range(len(h_list) - 1, -1, -1):
            h_t = h_list[t]
            h_prev = h_prev_list[t]
            x_t = x_list[t]

            d_z = d_h * (1.0 - h_t * h_t)
            d_W_xh += np.dot(x_t.T, d_z)
            d_W_hh += np.dot(h_prev.T, d_z)
            d_b_h += np.sum(d_z, axis=0, keepdims=True)
            d_h = np.dot(d_z, self.W_hh.T)

        grads = {
            "W_xh": d_W_xh,
            "W_hh": d_W_hh,
            "b_h": d_b_h,
            "W_hy": d_W_hy,
            "b_y": d_b_y,
        }
        return self._clip_grads(grads)

    def _update(self, grads: dict[str, np.ndarray]) -> None:
        beta1, beta2, eps = self._adam_hyperparams()
        self._adam_t += 1
        t = self._adam_t

        for name in self._param_names():
            grad = grads[name]
            m = self._adam_m[name]
            v = self._adam_v[name]

            m *= beta1
            m += (1.0 - beta1) * grad
            v *= beta2
            v += (1.0 - beta2) * (grad * grad)

            m_hat = m / (1.0 - beta1 ** t)
            v_hat = v / (1.0 - beta2 ** t)
            param = getattr(self, name)
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    @staticmethod
    def _batch_iter(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> tuple[np.ndarray, np.ndarray]:
        idx = np.arange(len(x))
        if shuffle:
            np.random.shuffle(idx)
        for start in range(0, len(x), batch_size):
            batch_idx = idx[start : start + batch_size]
            yield x[batch_idx], y[batch_idx]

    def predict_log(self, x: np.ndarray) -> np.ndarray:
        y_pred, _ = self._forward(x)
        return y_pred.reshape(-1)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> dict[str, list[float]]:
        best_val = float("inf")
        wait = 0
        best_state = None

        hist = {"train_log_mse": [], "val_log_mse": []}

        for epoch in range(1, self.max_epochs + 1):
            train_losses = []
            for xb, yb in self._batch_iter(x_train, y_train, self.batch_size, shuffle=True):
                y_pred, cache = self._forward(xb)
                loss = float(np.mean((y_pred.reshape(-1) - yb) ** 2))
                grads = self._backward(yb, cache)
                self._update(grads)
                train_losses.append(loss)

            train_loss = float(np.mean(train_losses)) if train_losses else 0.0
            val_pred = self.predict_log(x_val)
            val_loss = float(np.mean((val_pred - y_val) ** 2))

            hist["train_log_mse"].append(train_loss)
            hist["val_log_mse"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                wait = 0
                best_state = self._snapshot_state()
                print(f"Epoch {epoch:3d} | Train(log MSE): {train_loss:.4f} | Val(log MSE): {val_loss:.4f} | Best!")
            else:
                wait += 1
                print(f"Epoch {epoch:3d} | Train(log MSE): {train_loss:.4f} | Val(log MSE): {val_loss:.4f} | Wait: {wait}/{self.patience}")
                if wait >= self.patience:
                    print(f"Early stopping at epoch {epoch} (patience {self.patience} reached)")
                    break

        if best_state is not None:
            self._restore_state(best_state)

        return hist


def evaluate_cnt_metrics(model: SimpleRNNRegressor, x_seq: np.ndarray, y_cnt: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    pred_log = model.predict_log(x_seq)
    pred_cnt = np.expm1(pred_log)

    err = pred_cnt - y_cnt
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    denom = max(float(y_cnt.max() - y_cnt.min()), 1e-8) ** 2
    nmse = float(np.clip(mse / denom, 0.0, 1.0))
    return mse, mae, nmse, pred_cnt


def plot_hist(train_log: list[float], val_log: list[float], y_true: np.ndarray, y_pred: np.ndarray, out: Path) -> None:
    ep = np.arange(1, len(train_log) + 1)
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    ax[0].plot(ep, train_log, label="Train log-MSE")
    ax[0].plot(ep, val_log, label="Val log-MSE")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    n = min(250, len(y_true))
    ax[1].plot(np.arange(n), y_true[:n], label="Actual", alpha=0.8)
    ax[1].plot(np.arange(n), y_pred[:n], label="Pred", alpha=0.8)
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("cnt")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_test_pred(y_true: np.ndarray, y_pred: np.ndarray, out: Path, max_points: int = 50) -> None:
    n = min(max_points, len(y_true))
    idx = np.linspace(0, len(y_true) - 1, n, dtype=int)
    y_true_plot = y_true[idx]
    y_pred_plot = y_pred[idx]

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(np.arange(n), y_true_plot, "o-", label="Actual cnt", linewidth=1.5)
    ax.plot(np.arange(n), y_pred_plot, "o-", label="Predicted cnt", linewidth=1.5)
    ax.set_title(f"Validation: Actual vs Predicted ({n} points)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("cnt")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def load_hour_sequences(csv_path: Path, seq_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path).sort_values(["dteday", "hr"]).reset_index(drop=True)
    df["cnt_next"] = df["cnt"].shift(-1)
    df = df.dropna().reset_index(drop=True)

    x = df[FEATURES].values.astype(np.float32)
    y_cnt = df["cnt_next"].values.astype(np.float32)

    split = int(len(df) * 0.8)
    x_train_raw, x_val_raw = x[:split], x[split:]
    y_train_cnt, y_val_cnt = y_cnt[:split], y_cnt[split:]

    scaler_x = StandardScaler()
    x_train = scaler_x.fit_transform(x_train_raw)
    x_val = scaler_x.transform(x_val_raw)

    y_train_log = np.log1p(y_train_cnt)
    y_val_log = np.log1p(y_val_cnt)

    x_train_seq, y_train_log_seq = make_seq(x_train, y_train_log, seq_len)
    x_val_seq, y_val_log_seq = make_seq(x_val, y_val_log, seq_len)
    _, y_train_cnt_seq = make_seq(x_train, y_train_cnt, seq_len)
    _, y_val_cnt_seq = make_seq(x_val, y_val_cnt, seq_len)

    return x_train_seq, y_train_log_seq, x_val_seq, y_val_log_seq, y_train_cnt_seq, y_val_cnt_seq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="hour.csv")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    base = Path(__file__).resolve().parent
    csv_path = Path(args.csv) if Path(args.csv).is_absolute() else base / args.csv

    x_train, y_train_log, x_val, y_val_log, y_train_cnt, y_val_cnt = load_hour_sequences(csv_path, args.seq_len)

    print(f"Samples: train={len(x_train)}, val={len(x_val)} | Seq len={args.seq_len}")

    rnn = SimpleRNNRegressor(
        input_size=x_train.shape[-1],
        hidden_size=args.hidden,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        random_state=args.seed,
    )

    hist = rnn.fit(x_train, y_train_log, x_val, y_val_log)

    tr_mse, tr_mae, tr_nmse, tr_pred = evaluate_cnt_metrics(rnn, x_train, y_train_cnt)
    va_mse, va_mae, va_nmse, va_pred = evaluate_cnt_metrics(rnn, x_val, y_val_cnt)

    print("\nFinal metrics in cnt-space:")
    print(f"Train MAE={tr_mae:.3f} | Train MSE={tr_mse:.3f} | Train NMSE(0..1)={tr_nmse:.5f}")
    print(f"Val   MAE={va_mae:.3f} | Val   MSE={va_mse:.3f} | Val   NMSE(0..1)={va_nmse:.5f}")

    out_test = base / "rnn_numpy_test_pred_vs_real.png"
    plot_test_pred(y_val_cnt, va_pred, out_test, max_points=50)
    print(f"Test prediction plot saved: {out_test}")


if __name__ == "__main__":
    main()

