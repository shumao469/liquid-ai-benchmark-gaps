# Liquid Neural Network Revision Code

This branch contains the scientific implementation for the npj Robotics revision of *Liquid neural networks for embodied intelligence: From continuous-time mechanisms to deployment benchmarks*. It includes the model families, simulated tracking task, offline sequence pipeline, analysis code, experiment design templates, and tests.

## Run the local smoke test

Use Python 3.10 or newer. Install a PyTorch build appropriate for your platform and, for GPU work, the installed CUDA driver.

```bash
python -m venv .venv
# Activate the new virtual environment before installing.
python -m pip install -e ".[test]"
python -m liquid_revision.cli smoke --config configs/closed_loop_smoke.yaml --output-dir runs/smoke
python -m pytest tests/test_controller.py tests/test_models.py tests/test_perturbations.py tests/test_offline_metrics.py
```

The point-mass smoke test checks plumbing and produces no trained weights. The small `closed_loop_pilot.yaml` configuration uses the Gymnasium MuJoCo Reacher-v5 plant for development. Neither configuration produces evidence for the confirmatory comparison.

## Experimental design

The confirmatory simulated tracking task uses a moving joint reference in Gymnasium Reacher-v5. A learned causal observer estimates the four joint states from masked proprioception and timing features; the same fixed PD controller turns that estimate into a torque for every observer. Learned weights remain fixed during evaluation. The registered design compares LSTM, ODE-RNN, fixed-time-constant CT-RNN, LTC, dense CfC, NCP-CfC and a degree-matched random-sparse CfC control. It specifies five training seeds and 50,000 optimizer steps per registered run, with selection by validation loss.

The offline pipeline supports UCI HAR and PhysioNet 2012 and keeps trained checkpoints fixed across missingness conditions. Obtain dataset inputs through their official distribution channels and verify them with the data preparation and manifest utilities before starting a new run.

## What this branch can reproduce

- `src/liquid_revision/` contains the model, environment, data, training and analysis implementations.
- `configs/` separates smoke and pilot configurations from the confirmatory design templates.
- `tests/` exercises scientific code and numerical and provenance contracts with synthetic or temporary fixtures.

The confirmatory templates are design references. In particular, the closed-loop template deliberately leaves GPU identity, trusted binary digest and custody locations unset. A confirmatory replication needs its own independently frozen runtime, data and split manifests, validation and checkpoint releases, and explicit test authorization. The original machine-specific runtime custody paths and evidence are not distributed. The release copies of the custody-path constants point to a local, excluded evidence directory; the validation gates remain in place and fail closed without their evidence.

This branch contains no raw records, predictions, checkpoints, unpublished figures or manuscripts, reviewer correspondence, or production audit logs. The confirmatory closed-loop test has not been run; training completion is separate from test evaluation.
