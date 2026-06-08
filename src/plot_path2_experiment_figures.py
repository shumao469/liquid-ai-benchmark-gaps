from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULT_DIR = Path("results")
FIG_DIR = Path("figures_path2")
FIG_DIR.mkdir(exist_ok=True)


def require_file(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return Path(path)


def format_task_name(x):
    if str(x).lower() == "har":
        return "HAR"
    if str(x).lower() == "physionet":
        return "PhysioNet 2012"
    if str(x).lower() == "occupancy":
        return "Occupancy"
    return str(x)


def load_summary():
    f = require_file(RESULT_DIR / "summary_mean_std.csv")
    df = pd.read_csv(f)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_degradation():
    f = require_file(RESULT_DIR / "degradation_from_missing0.csv")
    df = pd.read_csv(f)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_raw():
    f1 = RESULT_DIR / "all_runs_deduplicated.csv"
    f2 = RESULT_DIR / "all_runs.csv"
    if f1.exists():
        f = f1
    else:
        f = require_file(f2)
    df = pd.read_csv(f)
    df.columns = [c.strip() for c in df.columns]
    return df


def plot_efficiency_performance():
    """
    Fig. 1: efficiency-performance scatter.
    x: inference time
    y: main metric
    point size: parameter count
    one panel per task at missingness=0.
    """
    summary = load_summary()

    if "missing_level" not in summary.columns:
        raise ValueError("summary_mean_std.csv must contain missing_level")

    sub = summary[summary["missing_level"] == 0.0].copy()

    # Fallback column compatibility
    if "main_metric_mean" not in sub.columns:
        if "test_accuracy_mean" in sub.columns:
            sub["main_metric_mean"] = sub["test_accuracy_mean"]
        else:
            raise ValueError("Cannot find main_metric_mean or test_accuracy_mean")

    if "test_infer_ms_per_sample_mean" not in sub.columns:
        raise ValueError("Cannot find test_infer_ms_per_sample_mean")

    if "n_params_mean" not in sub.columns:
        raise ValueError("Cannot find n_params_mean")

    model_order = ["lstm", "ltc", "cfc", "ncp_ltc", "ncp_cfc"]

    for task in sorted(sub["task"].unique()):
        gtask = sub[sub["task"] == task].copy()
        gtask["model"] = pd.Categorical(gtask["model"], categories=model_order, ordered=True)
        gtask = gtask.sort_values("model")

        plt.figure(figsize=(5.8, 4.5))

        for _, r in gtask.iterrows():
            size = max(60, float(r["n_params_mean"]) / 120.0)
            plt.scatter(
                r["test_infer_ms_per_sample_mean"],
                r["main_metric_mean"],
                s=size,
                alpha=0.75,
            )
            plt.text(
                r["test_infer_ms_per_sample_mean"],
                r["main_metric_mean"],
                "  " + str(r["model"]).upper(),
                va="center",
                fontsize=9,
            )

        ylabel = "Main metric"
        if str(task).lower() == "physionet":
            ylabel = "AUROC / balanced accuracy"
        elif str(task).lower() in ["har", "occupancy"]:
            ylabel = "Accuracy"

        plt.xlabel("Inference time (ms/sample)")
        plt.ylabel(ylabel)
        plt.title(f"{format_task_name(task)}: efficiency–performance trade-off")
        plt.tight_layout()

        out = FIG_DIR / f"fig1_efficiency_performance_{task}.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print("Saved:", out)


def plot_missingness_sensitivity():
    """
    Fig. 2: missingness / irregularity sensitivity curve.
    x: missing level
    y: performance drop relative to missing=0.
    """
    degradation = load_degradation()

    required = ["task", "model", "missing_level", "absolute_drop", "relative_drop_percent"]
    for c in required:
        if c not in degradation.columns:
            raise ValueError(f"Missing column in degradation table: {c}")

    model_order = ["lstm", "ltc", "cfc", "ncp_ltc", "ncp_cfc"]

    for task in sorted(degradation["task"].unique()):
        gtask = degradation[degradation["task"] == task].copy()

        plt.figure(figsize=(5.8, 4.5))

        for model in model_order:
            g = gtask[gtask["model"] == model].sort_values("missing_level")
            if g.empty:
                continue
            plt.plot(
                g["missing_level"],
                g["absolute_drop"],
                marker="o",
                label=model.upper(),
            )

        ylabel = "Performance drop from missing=0"
        if str(task).lower() == "physionet":
            ylabel = "AUROC drop from missing=0"
        elif str(task).lower() in ["har", "occupancy"]:
            ylabel = "Accuracy drop from missing=0"

        plt.xlabel("Missingness / irregularity level")
        plt.ylabel(ylabel)
        plt.title(f"{format_task_name(task)}: sensitivity to missingness")
        plt.legend()
        plt.tight_layout()

        out = FIG_DIR / f"fig2_missingness_sensitivity_{task}.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print("Saved:", out)


def plot_main_metric_vs_missingness():
    """
    Supplementary: raw performance vs missingness.
    """
    summary = load_summary()

    if "main_metric_mean" not in summary.columns:
        if "test_accuracy_mean" in summary.columns:
            summary["main_metric_mean"] = summary["test_accuracy_mean"]
        else:
            raise ValueError("Cannot find main_metric_mean or test_accuracy_mean")

    if "main_metric_std" not in summary.columns:
        if "test_accuracy_std" in summary.columns:
            summary["main_metric_std"] = summary["test_accuracy_std"]
        else:
            summary["main_metric_std"] = 0.0

    model_order = ["lstm", "ltc", "cfc", "ncp_ltc", "ncp_cfc"]

    for task in sorted(summary["task"].unique()):
        gtask = summary[summary["task"] == task].copy()

        plt.figure(figsize=(5.8, 4.5))

        for model in model_order:
            g = gtask[gtask["model"] == model].sort_values("missing_level")
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

        ylabel = "Main metric"
        if str(task).lower() == "physionet":
            ylabel = "AUROC / balanced accuracy"
        elif str(task).lower() in ["har", "occupancy"]:
            ylabel = "Accuracy"

        plt.xlabel("Missingness / irregularity level")
        plt.ylabel(ylabel)
        plt.title(f"{format_task_name(task)}: performance across missingness levels")
        plt.legend()
        plt.tight_layout()

        out = FIG_DIR / f"supp_main_metric_vs_missingness_{task}.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print("Saved:", out)


def plot_train_time_and_params():
    """
    Supplementary: training time and parameter count at missing=0.
    """
    summary = load_summary()

    if "missing_level" in summary.columns:
        sub = summary[summary["missing_level"] == 0.0].copy()
    else:
        sub = summary.copy()

    model_order = ["lstm", "ltc", "cfc", "ncp_ltc", "ncp_cfc"]

    for task in sorted(sub["task"].unique()):
        gtask = sub[sub["task"] == task].copy()
        gtask["model"] = pd.Categorical(gtask["model"], categories=model_order, ordered=True)
        gtask = gtask.sort_values("model")

        if "train_time_sec_mean" in gtask.columns:
            plt.figure(figsize=(5.4, 4.0))
            plt.bar(gtask["model"].astype(str).str.upper(), gtask["train_time_sec_mean"])
            plt.xlabel("Model")
            plt.ylabel("Training time per run (s)")
            plt.title(f"{format_task_name(task)}: training time at missing=0")
            plt.tight_layout()
            out = FIG_DIR / f"supp_training_time_{task}.png"
            plt.savefig(out, dpi=300)
            plt.close()
            print("Saved:", out)

        if "n_params_mean" in gtask.columns:
            plt.figure(figsize=(5.4, 4.0))
            plt.bar(gtask["model"].astype(str).str.upper(), gtask["n_params_mean"])
            plt.xlabel("Model")
            plt.ylabel("Trainable parameters")
            plt.title(f"{format_task_name(task)}: parameter count")
            plt.tight_layout()
            out = FIG_DIR / f"supp_parameter_count_{task}.png"
            plt.savefig(out, dpi=300)
            plt.close()
            print("Saved:", out)


def main():
    plot_efficiency_performance()
    plot_missingness_sensitivity()
    plot_main_metric_vs_missingness()
    plot_train_time_and_params()
    print("\nAll experiment figures saved to:", FIG_DIR.resolve())


if __name__ == "__main__":
    main()
