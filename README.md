# Liquid neural networks for embodied intelligence

Code for *Liquid neural networks for embodied intelligence: From continuous-time mechanisms to feedback-loop benchmarks*. The repository includes the simulated reaching task, recurrent observers, offline sequence pipeline, and analysis tools.

## Reproduce the 3,040 closed-loop rollouts

The two-link arm from Gymnasium's MuJoCo Reacher-v5 tracks a seeded moving target. Each learned observer estimates joint state from a common 19-channel causal input. A shared proportional–derivative controller uses those estimates to drive the arm. The evaluation pairs 20 trajectories across nominal sensing, 30% independent packet loss, burst loss, and four-step observation delay.

### 1. Install and choose a configuration

Use Python 3.10 or later and a CUDA-enabled PyTorch installation for the GPU workflow. Install the package from this directory and list the available devices.

```bash
python -m pip install -e ".[test]"
nvidia-smi -L
```

Set the configuration and run directory in a POSIX shell.

```bash
CFG=configs/closed_loop_paper.yaml
RUN=runs/paper
```

Set the same values in PowerShell.

```powershell
$CFG = "configs/closed_loop_paper.yaml"
$RUN = "runs/paper"
```

For a small end-to-end check, set `CFG` to `configs/closed_loop_paper_smoke.yaml` and `RUN` to `runs/paper-smoke`. The smoke configuration uses one final training seed, one evaluation trajectory and three optimizer updates per fit. It produces 40 rollouts per cohort, with 12 recomputed and 28 reused in the channel-scaled cohort.

### 2. Generate the demonstrations and parameter registry

```bash
python -m liquid_revision.cli closed-loop-prepare --config "$CFG" --output-dir "$RUN/data"
python -m liquid_revision.cli closed-loop-match --config "$CFG" --output "$RUN/registry.json"
```

The paper configuration generates 2,000 training and 200 validation demonstrations with disjoint trajectory and schedule seeds. The registry matches the seven learned observers to 20,000 active trainable parameters within 5%. The preparation stage materializes the training and validation data.

### 3. Tune the unit-range and channel-scaled observers

```bash
python -m liquid_revision.reproduction.tuning run --config "$CFG" --data-root "$RUN/data" --registry "$RUN/registry.json" --output-root "$RUN/tuning" --gpus 0,1 --workers-per-gpu 1 --readout unit
python -m liquid_revision.reproduction.tuning run --config "$CFG" --data-root "$RUN/data" --registry "$RUN/registry.json" --output-root "$RUN/tuning" --gpus 0,1 --workers-per-gpu 1 --readout scaled
```

Use the GPU indices for your machine. The launcher assigns each candidate process to one GPU and applies the per-GPU worker limit. The grid crosses learning rates `0.0003` and `0.001` with weight decays `0` and `0.0001`. The paper design uses seed 17 and 10,000 AdamW updates per candidate. Validation loss selects one pair for each observer and readout. The unit-range grid contains 28 candidates across seven observers. The channel-scaled grid contains 12 across dense CfC, NCP-CfC, and degree-matched random-sparse CfC. The selection files appear at `tuning/unit/tuning_selection.json` and `tuning/scaled/tuning_selection.json` within the run directory. Repeat the same command and output root to continue an interrupted grid.

### 4. Fit the selected observers

```bash
python -m liquid_revision.reproduction.training --config "$CFG" --data-root "$RUN/data" --registry "$RUN/registry.json" --selection "$RUN/tuning/unit/tuning_selection.json" --readout unit --output-dir "$RUN/fits" --device cuda:0
python -m liquid_revision.reproduction.training --config "$CFG" --data-root "$RUN/data" --registry "$RUN/registry.json" --selection "$RUN/tuning/scaled/tuning_selection.json" --readout scaled --output-dir "$RUN/fits" --device cuda:0
```

The paper design fits five seeds per observer at 50,000 optimizer updates each. The unit-range stage fits 35 checkpoints and the channel-scaled stage fits 15. The channel-scaled readout multiplies each bounded motor output by 1.05 times the maximum absolute standardized training target in that channel. The fixed scale is computed from the training split and is saved with the checkpoint identity.

### 5. Evaluate the paired cohorts and summarize

```bash
python -m liquid_revision.reproduction.evaluation --config "$CFG" --registry "$RUN/registry.json" --training-dir "$RUN/fits" --phase original --output-dir "$RUN/original" --device cuda:0
python -m liquid_revision.reproduction.evaluation --config "$CFG" --registry "$RUN/registry.json" --training-dir "$RUN/fits" --phase corrected --reuse-original "$RUN/original" --output-dir "$RUN/corrected" --device cuda:0
```

The unit-range run evaluates seven learned observers with five seeds each and three reference controllers. The channel-scaled run recomputes the three CfC families on the same trajectories and schedules. It combines **1,200** recomputed rows (3 observers × 5 seeds × 20 trajectories × 4 conditions) with **1,840** verified rows from the four other observers and the three reference controllers. The resulting **3,040** rows form the exploratory comparison cohort. Each full rollout covers up to 500 control steps, with a 1 m distance penalty for steps remaining after a registered failure.

In the corrected output directory, `episode_metrics.csv` contains the episode-level results. `mini_summary.json` reports tracking error, failures, and paired contrasts with descriptive percentile intervals from 10,000 resamples of training seeds and trajectories in the paper configuration. Retain both cohort manifests with their checkpoints and record the runtime and seed configuration. This workflow evaluates the exploratory cohort. The confirmatory track uses its own frozen design and test release.

## Additional workflows

- `configs/closed_loop_smoke.yaml` provides a point-mass check of the environment, data and model interfaces. Run `python -m liquid_revision.cli smoke --config configs/closed_loop_smoke.yaml --output-dir runs/smoke`.
- `configs/closed_loop_pilot.yaml` provides a small Reacher-v5 training and evaluation example. Run `python -m liquid_revision.cli closed-loop-pilot --config configs/closed_loop_pilot.yaml --output-dir runs/pilot --device cpu`.
- The confirmatory templates describe an independently frozen replication with its own runtime, data, split manifests, checkpoint releases, and test authorization.
- The offline pipeline supports UCI HAR and PhysioNet 2012. Obtain inputs through the official dataset distributors and verify them with the data preparation and manifest utilities.

Run `python -m pytest` for the test suite. Experiment outputs are written to local run directories.
