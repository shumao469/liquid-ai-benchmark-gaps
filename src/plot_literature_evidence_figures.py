from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


IN_FILE = Path("literature_coding/literature_evidence_coding.csv")
FIG_DIR = Path("figures_literature_evidence")
FIG_DIR.mkdir(exist_ok=True)


def load_data():
    if not IN_FILE.exists():
        raise FileNotFoundError(
            f"Missing {IN_FILE}. Run: python make_literature_coding_tables.py"
        )
    df = pd.read_csv(IN_FILE)
    df.columns = [c.strip() for c in df.columns]
    return df


def annotate_matrix(ax, data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            txt = "" if pd.isna(val) else str(int(val))
            ax.text(j, i, txt, ha="center", va="center", fontsize=9)


def plot_evidence_type_heatmap(df):
    """
    图 3：任务域证据地图热图
    domain × evidence_type count.
    """
    pivot = pd.crosstab(df["domain"], df["evidence_type"])

    domains = list(pivot.index)
    evidence_types = list(pivot.columns)
    data = pivot.values.astype(float)

    plt.figure(figsize=(max(7, 0.7 * len(evidence_types)), max(4, 0.5 * len(domains))))
    ax = plt.gca()
    im = ax.imshow(data, aspect="auto")

    ax.set_xticks(np.arange(len(evidence_types)))
    ax.set_xticklabels(evidence_types, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(domains)))
    ax.set_yticklabels(domains)

    ax.set_xlabel("Evidence type")
    ax.set_ylabel("Domain")
    ax.set_title("Evidence map across domains and evidence types")

    annotate_matrix(ax, data)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    out = FIG_DIR / "fig3_domain_evidence_type_heatmap.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)


def plot_reproducibility_matrix(df):
    """
    图 4：公开资源可复现性矩阵
    rows: study
    cols: public code/data/real-world/OOD/multimodal.
    """
    cols = [
        "public_code",
        "public_data",
        "real_world_validation",
        "multimodal",
        "direct_ood_evaluation",
    ]

    # One row per study: max over duplicate evidence rows
    mat = (
        df.groupby("study")[cols]
        .max()
        .reset_index()
        .sort_values("study")
    )

    studies = list(mat["study"])
    data = mat[cols].values.astype(float)

    plt.figure(figsize=(7.5, max(4, 0.35 * len(studies))))
    ax = plt.gca()
    im = ax.imshow(data, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(
        [
            "Public code",
            "Public data",
            "Real-world\nvalidation",
            "Multimodal",
            "Direct OOD\nevaluation",
        ],
        rotation=45,
        ha="right",
    )
    ax.set_yticks(np.arange(len(studies)))
    ax.set_yticklabels(studies)

    ax.set_title("Public-resource reproducibility matrix")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, "Yes" if data[i, j] == 1 else "No", ha="center", va="center", fontsize=8)

    plt.tight_layout()

    out = FIG_DIR / "fig4_reproducibility_matrix.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)


def plot_direct_adjacent_stacked_bar(df):
    """
    图 5：direct vs adjacent vs broader evidence stacked bar by domain.
    """
    count = (
        df.groupby(["domain", "evidence_category"])
        .size()
        .reset_index(name="n")
    )

    pivot = count.pivot(index="domain", columns="evidence_category", values="n").fillna(0)

    categories = ["direct", "adjacent", "broader background"]
    for c in categories:
        if c not in pivot.columns:
            pivot[c] = 0

    pivot = pivot[categories]
    domains = list(pivot.index)

    x = np.arange(len(domains))
    bottom = np.zeros(len(domains))

    plt.figure(figsize=(8, 4.5))

    for c in categories:
        vals = pivot[c].values
        plt.bar(x, vals, bottom=bottom, label=c)
        bottom += vals

    plt.xticks(x, domains, rotation=30, ha="right")
    plt.ylabel("Number of coded evidence items")
    plt.xlabel("Domain")
    plt.title("Direct vs adjacent vs broader evidence by domain")
    plt.legend()
    plt.tight_layout()

    out = FIG_DIR / "fig5_direct_adjacent_evidence_stacked_bar.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)


def plot_model_family_by_domain(df):
    """
    Optional: model family × domain heatmap.
    """
    pivot = pd.crosstab(df["domain"], df["model_family"])

    domains = list(pivot.index)
    families = list(pivot.columns)
    data = pivot.values.astype(float)

    plt.figure(figsize=(max(6, 0.7 * len(families)), max(4, 0.5 * len(domains))))
    ax = plt.gca()
    im = ax.imshow(data, aspect="auto")

    ax.set_xticks(np.arange(len(families)))
    ax.set_xticklabels(families, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(domains)))
    ax.set_yticklabels(domains)

    ax.set_xlabel("Model family / resource")
    ax.set_ylabel("Domain")
    ax.set_title("Model-family evidence coverage by domain")

    annotate_matrix(ax, data)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    out = FIG_DIR / "supp_model_family_by_domain_heatmap.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)


def main():
    df = load_data()

    plot_evidence_type_heatmap(df)
    plot_reproducibility_matrix(df)
    plot_direct_adjacent_stacked_bar(df)
    plot_model_family_by_domain(df)

    print("\nAll literature evidence figures saved to:", FIG_DIR.resolve())


if __name__ == "__main__":
    main()
