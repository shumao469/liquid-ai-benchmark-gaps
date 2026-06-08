# Liquid AI benchmark gaps

This repository supports the Comment **"Liquid AI needs sharper evidence, not broader claims"**.

The repository contains code, derived results and figure-generation scripts for a lightweight public-data sanity check comparing LSTM, liquid time-constant networks (LTCs) and closed-form continuous-time networks (CfCs) on regular and irregular time-series tasks.

## Purpose

The benchmark is intended as an illustrative reproducibility check, not as a definitive model ranking. It was used to support Fig. 1b–e of the Comment, which argues that liquid and continuous-time models should be evaluated jointly for performance, efficiency and degradation under missingness.

## Tasks

- HAR: regular time-series classification.
- PhysioNet 2012: irregular clinical time-series prediction.

Raw public datasets are not redistributed in this repository. Users should obtain them from their original public sources or through the dataset-specific access mechanisms.

## Models

- LSTM baseline
- LTC
- CfC

## Main parameters

- Random seeds: 1, 2, 3
- Synthetic missingness levels: 0.0, 0.3, 0.6
- Epochs: 50
- Hidden dimension: 64
- Batch size: 64
- Optimizer: Adam
- Learning rate: 1e-3

## Main outputs

- `results/all_runs_deduplicated.csv`
- `results/summary_mean_std.csv`
- `results/degradation_from_missing0.csv`
- `figures_path2/fig1_efficiency_performance_physionet.png`
- `figures_path2/fig2_missingness_sensitivity_physionet.png`
- `figures/physionet_main_metric_vs_missingness.png`

## Reproducing the figures

```bash
conda env create -f environment.yml
conda activate ltc_cfc_benchmark

bash scripts/01_run_benchmark.sh
bash scripts/02_summarize_results.sh
bash scripts/03_make_figures.sh
