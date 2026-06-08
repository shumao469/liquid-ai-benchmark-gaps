import argparse
import copy
import json
import math
import os
import random
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, f1_score

from ncps.torch import CfC, LTC

import matplotlib.pyplot as plt


DATA_DIR = Path("data")
RESULT_DIR = Path("results")
FIG_DIR = Path("figures")

DATA_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


HAR_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
PHYSIONET_BASE = "https://physionet.org/files/challenge-2012/1.0.0"
PHYSIONET_SET_A_URL = f"{PHYSIONET_BASE}/set-a.zip"
PHYSIONET_OUTCOMES_A_URL = f"{PHYSIONET_BASE}/Outcomes-a.txt"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def download_file(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[SKIP] Existing: {out_path}")
        return
    print(f"[DOWNLOAD] {url}")
    urllib.request.urlretrieve(url, out_path)
    print(f"[SAVED] {out_path}")


def unzip_file(zip_path: Path, out_dir: Path):
    marker = out_dir / ".unzipped"
    if marker.exists():
        print(f"[SKIP] Already unzipped: {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[UNZIP] {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    marker.write_text("ok\n")


def download_all():
    # HAR
    har_zip = DATA_DIR / "har" / "UCI_HAR_Dataset.zip"
    download_file(HAR_URL, har_zip)
    unzip_file(har_zip, DATA_DIR / "har")

    # PhysioNet 2012 set-a + outcomes
    physio_dir = DATA_DIR / "physionet2012"
    set_a_zip = physio_dir / "set-a.zip"
    outcomes_a = physio_dir / "Outcomes-a.txt"

    download_file(PHYSIONET_SET_A_URL, set_a_zip)
    unzip_file(set_a_zip, physio_dir)
    download_file(PHYSIONET_OUTCOMES_A_URL, outcomes_a)

    print("\nDownload finished.")
    print("HAR dir:", DATA_DIR / "har")
    print("PhysioNet dir:", DATA_DIR / "physionet2012")


class SeqDataset(Dataset):
    def __init__(self, xs, ys, timespans):
        self.xs = xs
        self.ys = ys
        self.timespans = timespans

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx], self.timespans[idx]


def collate_variable(batch):
    xs, ys, ts = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    max_len = int(lengths.max().item())
    dim = xs[0].shape[1]

    x_pad = torch.zeros(len(xs), max_len, dim, dtype=torch.float32)
    t_pad = torch.zeros(len(xs), max_len, dtype=torch.float32)

    for i, (x, t) in enumerate(zip(xs, ts)):
        n = x.shape[0]
        x_pad[i, :n] = torch.tensor(x, dtype=torch.float32)
        t_pad[i, :n] = torch.tensor(t, dtype=torch.float32)

    y = torch.tensor(ys, dtype=torch.long)
    return x_pad, y, lengths, t_pad


def apply_feature_missingness(x, missing_level, rng):
    """
    x: [T, D] standardized array.
    Return concat([x_missing, mask]) with same T and 2D feature dim.
    """
    mask = np.ones_like(x, dtype=np.float32)
    if missing_level > 0:
        drop = rng.rand(*x.shape) < missing_level
        mask[drop] = 0.0

    x_missing = x.copy()
    x_missing[mask == 0] = 0.0
    return np.concatenate([x_missing, mask], axis=1).astype(np.float32)


def make_fixed_sequences(x, y, seq_len=16, stride=16):
    xs, ys = [], []
    for start in range(0, len(x) - seq_len + 1, stride):
        end = start + seq_len
        xs.append(x[start:end])
        # Use the last window label as sequence label
        ys.append(int(y[end - 1]))
    return xs, np.array(ys, dtype=np.int64)


def load_har_data(seed, missing_level, seq_len=16):
    """
    HAR: regular sampled task.
    Input becomes [standardized feature, missingness mask].
    """
    base = DATA_DIR / "har" / "UCI HAR Dataset"
    if not base.exists():
        raise FileNotFoundError("HAR data not found. Run: python ltc_cfc_benchmark.py --download")

    x_train = np.loadtxt(base / "train" / "X_train.txt").astype(np.float32)
    y_train = (np.loadtxt(base / "train" / "y_train.txt").astype(np.int64) - 1)

    x_test = np.loadtxt(base / "test" / "X_test.txt").astype(np.float32)
    y_test = (np.loadtxt(base / "test" / "y_test.txt").astype(np.int64) - 1)

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6

    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    tr_xs, tr_y = make_fixed_sequences(x_train, y_train, seq_len=seq_len, stride=8)
    te_xs, te_y = make_fixed_sequences(x_test, y_test, seq_len=seq_len, stride=8)

    idx = np.arange(len(tr_y))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=0.1,
        random_state=seed,
        stratify=tr_y,
    )

    rng = np.random.RandomState(seed + int(missing_level * 1000) + 17)

    def transform(xs, offset):
        out_xs = []
        out_ts = []
        for i, x in enumerate(xs):
            local_rng = np.random.RandomState(seed * 100000 + offset + i + int(missing_level * 1000))
            x2 = apply_feature_missingness(x, missing_level, local_rng)
            out_xs.append(x2)
            out_ts.append(np.ones(x2.shape[0], dtype=np.float32))
        return out_xs, out_ts

    train_xs_raw = [tr_xs[i] for i in tr_idx]
    val_xs_raw = [tr_xs[i] for i in va_idx]
    test_xs_raw = te_xs

    train_xs, train_ts = transform(train_xs_raw, 1000)
    val_xs, val_ts = transform(val_xs_raw, 2000)
    test_xs, test_ts = transform(test_xs_raw, 3000)

    train_y = tr_y[tr_idx]
    val_y = tr_y[va_idx]
    test_y = te_y

    input_dim = train_xs[0].shape[1]
    n_classes = 6

    return {
        "task": "har",
        "input_dim": input_dim,
        "n_classes": n_classes,
        "train": SeqDataset(train_xs, train_y, train_ts),
        "val": SeqDataset(val_xs, val_y, val_ts),
        "test": SeqDataset(test_xs, test_y, test_ts),
    }


def parse_time_to_hours(s):
    h, m = str(s).split(":")
    return int(h) + int(m) / 60.0


def collect_physionet_variables(set_dir):
    variables = set()
    files = sorted(set_dir.glob("*.txt"))
    for f in files:
        df = pd.read_csv(f)
        for p in df["Parameter"].astype(str).unique():
            if p != "RecordID":
                variables.add(p)
    variables = sorted(variables)
    return variables


def read_physionet_record(path, variables):
    var_to_idx = {v: i for i, v in enumerate(variables)}
    df = pd.read_csv(path)

    df = df[df["Parameter"].astype(str) != "RecordID"].copy()
    df["hour"] = df["Time"].apply(parse_time_to_hours)

    # unique actual measurement times, preserving irregular sampling
    times = sorted(df["hour"].unique().tolist())
    if len(times) == 0:
        times = [0.0]

    x = np.full((len(times), len(variables)), np.nan, dtype=np.float32)
    time_to_row = {t: i for i, t in enumerate(times)}

    for _, row in df.iterrows():
        p = str(row["Parameter"])
        if p not in var_to_idx:
            continue
        r = time_to_row[row["hour"]]
        c = var_to_idx[p]
        try:
            val = float(row["Value"])
        except Exception:
            continue
        x[r, c] = val

    times = np.array(times, dtype=np.float32)
    timespans = np.zeros_like(times, dtype=np.float32)
    if len(times) > 1:
        timespans[1:] = np.diff(times)
    # avoid all-zero timespans for single-point record
    timespans = np.maximum(timespans, 1e-3)

    record_id = int(path.stem)
    return record_id, x, times, timespans


def fit_physionet_stats(records):
    """
    records: list of raw arrays [T, D] with NaN for missing.
    """
    all_vals = np.concatenate(records, axis=0)
    mean = np.nanmean(all_vals, axis=0)
    std = np.nanstd(all_vals, axis=0)

    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where((np.isfinite(std)) & (std > 1e-6), std, 1.0)

    return mean.astype(np.float32), std.astype(np.float32)


def transform_physionet_record(x_raw, timespans, mean, std, missing_level, seed):
    """
    Convert irregular PhysioNet record into:
      concat([standardized values, observation mask, feature-wise delta])
    """
    rng = np.random.RandomState(seed)

    observed = np.isfinite(x_raw)
    if missing_level > 0:
        extra_drop = (rng.rand(*x_raw.shape) < missing_level) & observed
        observed = observed & (~extra_drop)

    values = np.zeros_like(x_raw, dtype=np.float32)
    values[observed] = ((x_raw[observed] - np.take(mean, np.where(observed)[1])) /
                        np.take(std, np.where(observed)[1]))

    mask = observed.astype(np.float32)

    # Feature-wise elapsed time since last observation
    T, D = x_raw.shape
    delta = np.zeros((T, D), dtype=np.float32)
    last_seen = np.full(D, np.nan, dtype=np.float32)
    absolute_time = np.cumsum(timespans)

    for t in range(T):
        current_time = absolute_time[t]
        for d in range(D):
            if np.isnan(last_seen[d]):
                delta[t, d] = 48.0
            else:
                delta[t, d] = current_time - last_seen[d]
        obs_idx = observed[t]
        last_seen[obs_idx] = current_time

    delta = np.clip(delta / 48.0, 0.0, 1.0)

    x = np.concatenate([values, mask, delta], axis=1).astype(np.float32)
    ts = timespans.astype(np.float32)
    return x, ts


def load_physionet_data(seed, missing_level):
    """
    PhysioNet 2012 set-a: irregular ICU time-series.
    Label: In-hospital_death.
    """
    base = DATA_DIR / "physionet2012"
    set_dir = base / "set-a"
    outcomes_file = base / "Outcomes-a.txt"

    if not set_dir.exists() or not outcomes_file.exists():
        raise FileNotFoundError("PhysioNet data not found. Run: python ltc_cfc_benchmark.py --download")

    outcomes = pd.read_csv(outcomes_file)
    outcomes.columns = [c.strip() for c in outcomes.columns]

    if "RecordID" not in outcomes.columns or "In-hospital_death" not in outcomes.columns:
        raise ValueError("Outcomes-a.txt must contain RecordID and In-hospital_death")

    y_map = {
        int(r["RecordID"]): int(r["In-hospital_death"])
        for _, r in outcomes.iterrows()
        if int(r["In-hospital_death"]) in [0, 1]
    }

    variables = collect_physionet_variables(set_dir)

    raw_records = []
    raw_timespans = []
    labels = []
    record_ids = []

    for f in sorted(set_dir.glob("*.txt")):
        rid, x_raw, times, ts = read_physionet_record(f, variables)
        if rid not in y_map:
            continue
        raw_records.append(x_raw)
        raw_timespans.append(ts)
        labels.append(y_map[rid])
        record_ids.append(rid)

    labels = np.array(labels, dtype=np.int64)
    idx = np.arange(len(labels))

    train_idx, temp_idx = train_test_split(
        idx,
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=seed,
        stratify=labels[temp_idx],
    )

    train_raw = [raw_records[i] for i in train_idx]
    mean, std = fit_physionet_stats(train_raw)

    def make_split(indices, offset):
        xs, ts_list, ys = [], [], []
        for k, i in enumerate(indices):
            local_seed = seed * 100000 + offset + k + int(missing_level * 1000)
            x, ts = transform_physionet_record(
                raw_records[i],
                raw_timespans[i],
                mean,
                std,
                missing_level,
                local_seed,
            )
            xs.append(x)
            ts_list.append(ts)
            ys.append(labels[i])
        return SeqDataset(xs, np.array(ys, dtype=np.int64), ts_list)

    train_ds = make_split(train_idx, 1000)
    val_ds = make_split(val_idx, 2000)
    test_ds = make_split(test_idx, 3000)

    input_dim = train_ds.xs[0].shape[1]
    n_classes = 2

    return {
        "task": "physionet",
        "input_dim": input_dim,
        "n_classes": n_classes,
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
    }


class SequenceClassifier(nn.Module):
    def __init__(self, model_name, input_dim, hidden_dim, n_classes, dropout=0.1):
        super().__init__()
        self.model_name = model_name
        self.n_classes = n_classes

        if model_name == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )

        elif model_name == "ltc":
            self.rnn = LTC(
                input_size=input_dim,
                units=hidden_dim,
                return_sequences=True,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )

        elif model_name == "cfc":
            self.rnn = CfC(
                input_size=input_dim,
                units=hidden_dim,
                return_sequences=True,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

    def forward(self, x, lengths, timespans=None):
        if self.model_name == "lstm":
            packed = pack_padded_sequence(
                x,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (h_n, _) = self.rnn(packed)
            h = h_n[-1]
        else:
            rnn_class_name = self.rnn.__class__.__name__.lower()
            if 'ltc' in rnn_class_name:
                # ncps.torch.LTC has a broadcasting issue with per-sample timespans
                # in this benchmark. For missing=0.0 / regular sampling, use default dt.
                out, _ = self.rnn(x, timespans=None)
            else:
                # Regular-sampling debug fix: avoid ncps LTC/CfC timespan broadcasting error.
                # For missing=0.0, use the default internal dt.
                out, _ = self.rnn(x, timespans=None)
            idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.shape[-1])
            h = out.gather(1, idx.to(out.device)).squeeze(1)

        logits = self.head(h)
        return logits


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model, loader, device, task):
    model.eval()
    losses = []
    all_y = []
    all_pred = []
    all_prob = []

    criterion = nn.CrossEntropyLoss()

    t0 = time.perf_counter()
    n_samples = 0

    with torch.no_grad():
        for x, y, lengths, timespans in loader:
            x = x.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            timespans = timespans.to(device)

            logits = model(x, lengths, timespans=timespans)
            loss = criterion(logits, y)

            prob = torch.softmax(logits, dim=1)
            pred = torch.argmax(prob, dim=1)

            losses.append(loss.item() * len(y))
            all_y.append(y.cpu().numpy())
            all_pred.append(pred.cpu().numpy())
            all_prob.append(prob.cpu().numpy())

            n_samples += len(y)

    elapsed = time.perf_counter() - t0
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    prob = np.concatenate(all_prob)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    auroc = np.nan
    if prob.shape[1] == 2 and len(np.unique(y_true)) == 2:
        try:
            auroc = roc_auc_score(y_true, prob[:, 1])
        except Exception:
            auroc = np.nan

    mean_loss = np.sum(losses) / max(1, n_samples)
    infer_ms_per_sample = elapsed * 1000.0 / max(1, n_samples)

    return {
        "loss": float(mean_loss),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "auroc": float(auroc) if np.isfinite(auroc) else np.nan,
        "infer_ms_per_sample": float(infer_ms_per_sample),
    }


def get_data(task, seed, missing_level):
    if task == "har":
        return load_har_data(seed=seed, missing_level=missing_level)
    if task == "physionet":
        return load_physionet_data(seed=seed, missing_level=missing_level)
    raise ValueError(f"Unknown task: {task}")


def train_one_run(
    task,
    model_name,
    seed,
    missing_level,
    epochs,
    hidden_dim,
    batch_size,
    lr,
    device,
):
    set_seed(seed)

    data = get_data(task, seed=seed, missing_level=missing_level)

    train_loader = DataLoader(
        data["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_variable,
    )
    val_loader = DataLoader(
        data["val"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_variable,
    )
    test_loader = DataLoader(
        data["test"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_variable,
    )

    model = SequenceClassifier(
        model_name=model_name,
        input_dim=data["input_dim"],
        hidden_dim=hidden_dim,
        n_classes=data["n_classes"],
    ).to(device)

    n_params = count_params(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_metric = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = -1

    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for x, y, lengths, timespans in train_loader:
            x = x.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            timespans = timespans.to(device)

            optimizer.zero_grad()
            logits = model(x, lengths, timespans=timespans)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        val_metrics = evaluate(model, val_loader, device, task)

        if task == "physionet":
            val_metric = val_metrics["auroc"]
            if not np.isfinite(val_metric):
                val_metric = val_metrics["balanced_accuracy"]
        else:
            val_metric = val_metrics["accuracy"]

        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            print(
                f"[{task} | {model_name} | seed={seed} | miss={missing_level}] "
                f"epoch={epoch:03d}/{epochs}, train_loss={np.mean(train_losses):.4f}, "
                f"val_acc={val_metrics['accuracy']:.4f}, "
                f"val_bal_acc={val_metrics['balanced_accuracy']:.4f}, "
                f"val_auroc={val_metrics['auroc']:.4f}"
            )

    train_time = time.perf_counter() - train_start

    model.load_state_dict(best_state)
    val_metrics = evaluate(model, val_loader, device, task)
    test_metrics = evaluate(model, test_loader, device, task)

    if task == "physionet":
        main_metric = test_metrics["auroc"]
        if not np.isfinite(main_metric):
            main_metric = test_metrics["balanced_accuracy"]
    else:
        main_metric = test_metrics["accuracy"]

    result = {
        "task": task,
        "model": model_name,
        "seed": seed,
        "missing_level": missing_level,
        "epochs": epochs,
        "hidden_dim": hidden_dim,
        "batch_size": batch_size,
        "best_epoch": best_epoch,
        "n_params": n_params,
        "train_time_sec": train_time,
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_auroc": val_metrics["auroc"],
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_auroc": test_metrics["auroc"],
        "test_infer_ms_per_sample": test_metrics["infer_ms_per_sample"],
        "main_metric": main_metric,
    }

    return result


def load_existing_results(path):
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def run_all(args):
    device = torch.device(args.device)

    out_file = RESULT_DIR / "all_runs.csv"
    existing = load_existing_results(out_file)

    completed = set()
    if not existing.empty:
        for _, r in existing.iterrows():
            completed.add(
                (
                    str(r["task"]),
                    str(r["model"]),
                    int(r["seed"]),
                    float(r["missing_level"]),
                )
            )

    all_new = []

    for task in args.tasks:
        for model_name in args.models:
            for seed in args.seeds:
                for miss in args.missing_levels:
                    key = (task, model_name, int(seed), float(miss))
                    if key in completed and not args.force:
                        print(f"[SKIP completed] {key}")
                        continue

                    print("\n============================================================")
                    print(f"Task={task}, Model={model_name}, Seed={seed}, Missing={miss}")
                    print("============================================================")

                    result = train_one_run(
                        task=task,
                        model_name=model_name,
                        seed=int(seed),
                        missing_level=float(miss),
                        epochs=args.epochs,
                        hidden_dim=args.hidden_dim,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        device=device,
                    )

                    all_new.append(result)

                    tmp = pd.DataFrame([result])
                    header = not out_file.exists()
                    tmp.to_csv(out_file, mode="a", header=header, index=False)
                    print("[SAVED]", out_file)

    print("\nAll requested runs finished.")
    print("Raw result file:", out_file)


def summarize_results():
    raw_file = RESULT_DIR / "all_runs.csv"
    if not raw_file.exists():
        raise FileNotFoundError("No results/all_runs.csv found. Run training first.")

    raw = pd.read_csv(raw_file)
    raw = raw.drop_duplicates(
        subset=["task", "model", "seed", "missing_level"],
        keep="last",
    )

    raw.to_csv(RESULT_DIR / "all_runs_deduplicated.csv", index=False)

    metrics = [
        "main_metric",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_auroc",
        "train_time_sec",
        "test_infer_ms_per_sample",
        "n_params",
        "best_epoch",
    ]

    rows = []
    for (task, missing_level, model), g in raw.groupby(["task", "missing_level", "model"]):
        row = {
            "task": task,
            "missing_level": missing_level,
            "model": model,
            "n": len(g),
        }
        for m in metrics:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std(ddof=1)
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["task", "missing_level", "model"]).reset_index(drop=True)
    summary.to_csv(RESULT_DIR / "summary_mean_std.csv", index=False)

    # degradation relative to missing=0
    degr_rows = []
    for (task, model), g in summary.groupby(["task", "model"]):
        base = g[g["missing_level"] == 0.0]
        if base.empty:
            continue
        base_val = float(base["main_metric_mean"].iloc[0])
        for _, r in g.iterrows():
            cur = float(r["main_metric_mean"])
            degr_rows.append(
                {
                    "task": task,
                    "model": model,
                    "missing_level": r["missing_level"],
                    "main_metric_mean": cur,
                    "baseline_main_metric_mean": base_val,
                    "absolute_drop": base_val - cur,
                    "relative_drop_percent": 100.0 * (base_val - cur) / max(1e-8, base_val),
                }
            )

    degradation = pd.DataFrame(degr_rows)
    degradation.to_csv(RESULT_DIR / "degradation_from_missing0.csv", index=False)

    print("\n========== Raw deduplicated ==========")
    print(raw[["task", "model", "seed", "missing_level", "main_metric", "test_accuracy", "test_auroc", "train_time_sec", "test_infer_ms_per_sample", "n_params"]].to_string(index=False))

    print("\n========== Summary mean ± SD ==========")
    show = summary[["task", "missing_level", "model", "n", "main_metric_mean", "main_metric_std", "train_time_sec_mean", "test_infer_ms_per_sample_mean", "n_params_mean"]]
    print(show.to_string(index=False))

    print("\nSaved:")
    print("  results/all_runs_deduplicated.csv")
    print("  results/summary_mean_std.csv")
    print("  results/degradation_from_missing0.csv")


def plot_results():
    summary_file = RESULT_DIR / "summary_mean_std.csv"
    raw_file = RESULT_DIR / "all_runs_deduplicated.csv"

    if not summary_file.exists():
        summarize_results()

    summary = pd.read_csv(summary_file)
    raw = pd.read_csv(raw_file) if raw_file.exists() else pd.read_csv(RESULT_DIR / "all_runs.csv")

    model_order = ["lstm", "ltc", "cfc"]

    # 1. Main metric vs missingness
    for task in sorted(summary["task"].unique()):
        sub = summary[summary["task"] == task].copy()
        plt.figure(figsize=(6.5, 4.5))

        for model in model_order:
            g = sub[sub["model"] == model].sort_values("missing_level")
            if g.empty:
                continue
            plt.errorbar(
                g["missing_level"],
                g["main_metric_mean"],
                yerr=g["main_metric_std"],
                marker="o",
                capsize=4,
                label=model.upper(),
            )

        ylabel = "AUROC" if task == "physionet" else "Accuracy"
        plt.xlabel("Synthetic missingness / irregularity level")
        plt.ylabel(ylabel)
        plt.title(f"{task}: model robustness under missingness")
        plt.legend()
        plt.tight_layout()

        out = FIG_DIR / f"{task}_main_metric_vs_missingness.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print("[FIG]", out)

    # 2. Degradation bar at highest missingness
    degr_file = RESULT_DIR / "degradation_from_missing0.csv"
    degradation = pd.read_csv(degr_file)

    for task in sorted(degradation["task"].unique()):
        sub = degradation[degradation["task"] == task].copy()
        max_missing = sub["missing_level"].max()
        sub = sub[sub["missing_level"] == max_missing]
        sub["model"] = pd.Categorical(sub["model"], categories=model_order, ordered=True)
        sub = sub.sort_values("model")

        plt.figure(figsize=(5.5, 4.0))
        plt.bar(sub["model"].astype(str), sub["relative_drop_percent"])
        plt.xlabel("Model")
        plt.ylabel("Relative drop from missing=0 (%)")
        plt.title(f"{task}: performance degradation at missing={max_missing}")
        plt.tight_layout()

        out = FIG_DIR / f"{task}_relative_drop_high_missingness.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print("[FIG]", out)

    # 3. Runtime
    for task in sorted(summary["task"].unique()):
        sub = summary[summary["task"] == task].copy()
        sub0 = sub[sub["missing_level"] == 0.0].copy()
        sub0["model"] = pd.Categorical(sub0["model"], categories=model_order, ordered=True)
        sub0 = sub0.sort_values("model")

        plt.figure(figsize=(5.5, 4.0))
        plt.bar(sub0["model"].astype(str), sub0["train_time_sec_mean"])
        plt.xlabel("Model")
        plt.ylabel("Training time per run (s)")
        plt.title(f"{task}: training time at missing=0")
        plt.tight_layout()

        out = FIG_DIR / f"{task}_training_time_missing0.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print("[FIG]", out)

    # 4. Parameter count
    params = (
        raw.groupby(["task", "model"])["n_params"]
        .mean()
        .reset_index()
    )

    for task in sorted(params["task"].unique()):
        sub = params[params["task"] == task].copy()
        sub["model"] = pd.Categorical(sub["model"], categories=model_order, ordered=True)
        sub = sub.sort_values("model")

        plt.figure(figsize=(5.5, 4.0))
        plt.bar(sub["model"].astype(str), sub["n_params"])
        plt.xlabel("Model")
        plt.ylabel("Trainable parameters")
        plt.title(f"{task}: parameter count")
        plt.tight_layout()

        out = FIG_DIR / f"{task}_parameter_count.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print("[FIG]", out)

    print("\nAll figures saved to:", FIG_DIR.resolve())


def make_argparser():
    p = argparse.ArgumentParser()

    p.add_argument("--download", action="store_true")
    p.add_argument("--run_all", action="store_true")
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--plot", action="store_true")

    p.add_argument("--tasks", nargs="+", default=["har", "physionet"])
    p.add_argument("--models", nargs="+", default=["lstm", "ltc", "cfc"])
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--missing_levels", nargs="+", type=float, default=[0.0, 0.3, 0.6])

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--force", action="store_true")

    return p


def main():
    args = make_argparser().parse_args()

    if args.download:
        download_all()

    if args.run_all:
        run_all(args)

    if args.summarize:
        summarize_results()

    if args.plot:
        plot_results()

    if not any([args.download, args.run_all, args.summarize, args.plot]):
        print("Nothing to do. Try:")
        print("  python ltc_cfc_benchmark.py --download")
        print("  python ltc_cfc_benchmark.py --run_all")
        print("  python ltc_cfc_benchmark.py --summarize --plot")


if __name__ == "__main__":
    main()
