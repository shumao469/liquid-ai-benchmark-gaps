"""Synthetic train-step timing; never opens a dataset or computes test metrics."""
from __future__ import annotations

import argparse
import statistics
import time
import types
from pathlib import Path

import torch

from ..config import load_config
from ..io_utils import atomic_write_json
from ..repro import seed_everything
from .formal import current_run_accelerator_identity, utc_now
from .models import build_offline_model, resolve_hidden_dim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--models", nargs="+", default=["ltc", "ode-rnn", "cfc", "lstm"])
    parser.add_argument("--measured-steps", type=int, default=5)
    parser.add_argument("--ode-sync-mode", choices=["current", "legacy"], default="current")
    args = parser.parse_args()
    config, config_hash = load_config(args.config)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_interop_threads(1)
    rows = []
    for threads in args.threads:
        torch.set_num_threads(threads)
        for name in args.models:
            seed_everything(17, True)
            model = build_offline_model(name, input_dim=28,
                hidden_dim=resolve_hidden_dim(config["models"], "har", name), output_dim=6,
                ode_max_step=1.0, ltc_ode_unfolds=6, cfc_mode="default").cuda()
            if name == "ode-rnn" and args.ode_sync_mode == "legacy":
                def legacy_forward(encoder, values, elapsed):
                    encoder.validate(values, elapsed, encoder.input_dim)
                    if elapsed.ndim == 3:
                        elapsed = elapsed[..., 0]
                    state = values.new_zeros(values.shape[0], encoder.hidden_dim)
                    outputs = []
                    for index in range(values.shape[1]):
                        state = encoder._evolve(state, elapsed[:, index])
                        state = encoder.observation_update(values[:, index], state)
                        outputs.append(state)
                    return torch.stack(outputs, dim=1)
                model.encoder.forward = types.MethodType(legacy_forward, model.encoder)
            # Match the expensive HAR training shape without touching any held-out arrays.
            features = torch.randn(64, 128, 28, device="cuda")
            dt = torch.full((64, 128), 0.02, device="cuda")
            lengths = torch.full((64,), 128, dtype=torch.long, device="cuda")
            labels = torch.arange(64, device="cuda") % 6
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
            times = []
            for step in range(2 + args.measured_steps):
                torch.cuda.synchronize()
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(features, dt, lengths), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                torch.cuda.synchronize()
                if step >= 2:
                    times.append(time.perf_counter() - started)
            rows.append({"dataset_shape": "har", "model": name, "cpu_threads": threads,
                         "median_step_seconds": statistics.median(times),
                         "step_seconds": times, "finite_loss": bool(torch.isfinite(loss))})
            del model, optimizer, features, dt, lengths, labels, loss
            torch.cuda.empty_cache()
    atomic_write_json(output, {"created_at_utc": utc_now(), "config_sha256": config_hash,
        "accelerator": current_run_accelerator_identity("cuda:0"),
        "precision": "float32", "autocast": False, "test_arrays_opened": False,
        "synthetic_profile_only": True, "ode_sync_mode": args.ode_sync_mode, "rows": rows})


if __name__ == "__main__":
    main()
