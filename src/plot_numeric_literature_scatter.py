from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


IN_FILE = Path("literature_coding/github_literature_numeric_results.csv")
OUT_DIR = Path("figures_literature_numeric")
OUT_DIR.mkdir(exist_ok=True)


def main():
    df = pd.read_csv(IN_FILE)

    for col in ["metric_value", "train_time_sec", "infer_time_ms", "params"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["metric_value", "infer_time_ms", "params"])

    for task in sorted(df["task"].unique()):
        sub = df[df["task"] == task].copy()

        plt.figure(figsize=(5.8, 4.5))

        for _, r in sub.iterrows():
            size = max(60, r["params"] / 120.0)
            plt.scatter(r["infer_time_ms"], r["metric_value"], s=size, alpha=0.75)
            plt.text(
                r["infer_time_ms"],
                r["metric_value"],
                "  " + str(r["model"]).upper(),
                va="center",
                fontsize=9,
            )

        plt.xlabel("Inference time")
        plt.ylabel("Reported performance")
        plt.title(f"{task}: reported efficiency–performance relation")
        plt.tight_layout()

        out = OUT_DIR / f"{task}_reported_efficiency_performance.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print("Saved:", out)

    print("\nDone.")


if __name__ == "__main__":
    main()
