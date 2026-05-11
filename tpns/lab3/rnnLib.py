from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


FEATURES = [
    "season", "yr", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "temp", "atemp", "hum", "windspeed",
]


class SeqDS(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class RNN(nn.Module):
    def __init__(self, n_feat, hid=128, layers=1, drop=0.2, model="rnn"):
        super().__init__()
        model = model.lower()
        if model not in {"rnn", "gru", "lstm"}:
            raise ValueError("model must be one of: rnn, gru, lstm")

        self.model = model
        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[model]
        self.rnn = rnn_cls(n_feat, hid, num_layers=layers, batch_first=True, dropout=drop if layers > 1 else 0.0)
        self.fc = nn.Linear(hid, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


def make_seq(x, y, seq_len):
    xs, ys = [], []
    for i in range(len(x) - seq_len + 1):
        xs.append(x[i:i + seq_len])
        ys.append(y[i + seq_len - 1])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def eval_cnt(model, loader, y_true_cnt, dev, inv_fn=np.expm1):
    model.eval()
    p = []
    with torch.no_grad():
        for xb, _ in loader:
            p.append(model(xb.to(dev)).squeeze(1).cpu().numpy())
    pred_cnt = inv_fn(np.concatenate(p))
    err = pred_cnt - y_true_cnt
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    nmse = float(np.clip(mse / (max(y_true_cnt.max() - y_true_cnt.min(), 1e-8) ** 2), 0.0, 1.0))
    return mse, mae, nmse, pred_cnt


def plot_hist(train_log, val_log, val_mse_cnt, y_true, y_pred, out):
    ep = np.arange(1, len(train_log) + 1)
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    ax[0].plot(ep, train_log, label="Train log-MSE")
    ax[0].plot(ep, val_log, label="Val log-MSE")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(ep, val_mse_cnt, label="Val MSE(cnt)")
    n = min(250, len(y_true))
    ax[1].plot(np.arange(n), y_true[:n], label="Actual", alpha=0.8)
    ax[1].plot(np.arange(n), y_pred[:n], label="Pred", alpha=0.8)
    ax[1].set_xlabel("Epoch / Time")
    ax[1].set_ylabel("cnt")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_test_pred(y_true, y_pred, out, max_points=50):
    n = min(max_points, len(y_true))
    idx = np.linspace(0, len(y_true) - 1, n, dtype=int)
    y_true_plot = y_true[idx]
    y_pred_plot = y_pred[idx]

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(np.arange(n), y_true_plot, "o-", label="Actual cnt", linewidth=1.5)
    ax.plot(np.arange(n), y_pred_plot, "o-", label="Predicted cnt", linewidth=1.5)
    ax.set_title(f"Test/Validation: Actual vs Predicted ({n} points)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("cnt")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="hour.csv")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--model", choices=["rnn", "gru", "lstm"], default="rnn")
    args = ap.parse_args()

    np.random.seed(43)
    torch.manual_seed(43)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = Path(__file__).resolve().parent
    csv = Path(args.csv) if Path(args.csv).is_absolute() else base / args.csv

    df = pd.read_csv(csv).sort_values(["dteday", "hr"]).reset_index(drop=True)
    df["cnt_next"] = df["cnt"].shift(-1)
    df = df.dropna().reset_index(drop=True)

    x = df[FEATURES].values.astype(np.float32)
    y = df["cnt_next"].values.astype(np.float32)
    split = int(len(df) * 0.8)

    sc = StandardScaler()
    x_tr = sc.fit_transform(x[:split])
    x_va = sc.transform(x[split:])

    y_tr_cnt, y_va_cnt = y[:split], y[split:]
    y_tr_log, y_va_log = np.log1p(y_tr_cnt), np.log1p(y_va_cnt)

    x_tr_s, y_tr_s = make_seq(x_tr, y_tr_log, args.seq_len)
    x_va_s, y_va_s = make_seq(x_va, y_va_log, args.seq_len)
    _, y_tr_s_cnt = make_seq(x_tr, y_tr_cnt, args.seq_len)
    _, y_va_s_cnt = make_seq(x_va, y_va_cnt, args.seq_len)

    inv_fn = np.expm1
    train_shuffle = True
    loss_space = "log"

    tr_loader = DataLoader(SeqDS(x_tr_s, y_tr_s), batch_size=args.batch_size, shuffle=train_shuffle)
    tr_eval_loader = DataLoader(SeqDS(x_tr_s, y_tr_s), batch_size=args.batch_size, shuffle=False)
    va_loader = DataLoader(SeqDS(x_va_s, y_va_s), batch_size=args.batch_size, shuffle=False)

    model = RNN(n_feat=x_tr_s.shape[-1], hid=args.hidden, model=args.model).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.MSELoss()

    best, best_state, wait = float("inf"), None, 0
    h_tr, h_va, h_mse = [], [], []

    print(f"Device: {dev} | Model: {args.model.upper()} | Samples: train={len(x_tr_s)}, val={len(x_va_s)}")
    for e in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(dev), yb.to(dev).unsqueeze(1)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        tr_loss /= max(len(tr_loader), 1)

        model.eval()
        p_val_log, t_val_log = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                p_val_log.append(model(xb.to(dev)).squeeze(1).cpu().numpy())
                t_val_log.append(yb.numpy())
        p_val_log = np.concatenate(p_val_log)
        t_val_log = np.concatenate(t_val_log)
        va_loss = float(np.mean((p_val_log - t_val_log) ** 2))

        train_mse_cnt, _, _, _ = eval_cnt(model, tr_eval_loader, y_tr_s_cnt, dev, inv_fn=inv_fn)
        val_mse_cnt, val_mae_cnt, val_nmse, val_pred_cnt = eval_cnt(model, va_loader, y_va_s_cnt, dev, inv_fn=inv_fn)

        h_tr.append(tr_loss)
        h_va.append(va_loss)
        h_mse.append(val_mse_cnt)

        print(
            f"Epoch {e:3d} | Train({loss_space} MSE): {tr_loss:.4f} | Train(cnt MSE): {train_mse_cnt:.1f} | "
            f"Val({loss_space} MSE): {va_loss:.4f} | Val MAE(cnt): {val_mae_cnt:.1f} | "
            f"Val MSE(cnt): {val_mse_cnt:.1f} | Val NMSE(0..1): {val_nmse:.4f}"
        )

        if va_loss < best:
            best, wait = va_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {e}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    v_mse, v_mae, v_nmse, v_pred = eval_cnt(model, va_loader, y_va_s_cnt, dev, inv_fn=inv_fn)
    print("\nFinal validation metrics (cnt):")
    print(f"MAE={v_mae:.3f} | MSE={v_mse:.3f} | RMSE={np.sqrt(v_mse):.3f} | NMSE(0..1)={v_nmse:.5f}")


    out_test = base / "rnn_test_pred_vs_real.png"
    plot_test_pred(y_va_s_cnt, v_pred, out_test)
    print(f"Test prediction plot saved: {out_test}")


if __name__ == "__main__":
    main()
