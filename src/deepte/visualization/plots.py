"""Publication-quality figure generators for DeepTE-Peptide."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

# Font setup (CJK + Latin fallback)
try:
    fm.fontManager.addfont("/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf")
    fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
except Exception:
    pass
plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# Nature Biotechnology-style color palette
NBT_PALETTE = {
    "primary": "#1F4E79",  # Deep blue
    "secondary": "#C0504D",  # Muted red
    "tertiary": "#9BBB59",  # Olive green
    "quaternary": "#8064A2",  # Purple
    "accent": "#4BACC6",  # Cyan
    "muted": "#A6A6A6",
    "background": "#FFFFFF",
}

OUTPUT_DPI = 300


def _save(fig, out_path: str | Path, dpi: int = OUTPUT_DPI) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# ROC / PR curves
# ---------------------------------------------------------------------------
def plot_roc_curves(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: str | Path,
    title: str = "ROC curves",
) -> Path:
    """Plot ROC curves for multiple methods.

    Parameters
    ----------
    results
        Mapping method_name -> (y_true, y_score).
    """
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

    colors = [
        NBT_PALETTE["primary"],
        NBT_PALETTE["secondary"],
        NBT_PALETTE["tertiary"],
        NBT_PALETTE["quaternary"],
        NBT_PALETTE["accent"],
        NBT_PALETTE["muted"],
    ]

    for (name, (y_true, y_score)), color in zip(results.items(), colors, strict=False):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = average_precision_score(y_true, y_score)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return _save(fig, out_path)


def plot_pr_curves(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: str | Path,
    title: str = "Precision-Recall curves",
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    colors = [
        NBT_PALETTE["primary"],
        NBT_PALETTE["secondary"],
        NBT_PALETTE["tertiary"],
        NBT_PALETTE["quaternary"],
        NBT_PALETTE["accent"],
        NBT_PALETTE["muted"],
    ]

    for (name, (y_true, y_score)), color in zip(results.items(), colors, strict=False):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, lw=2, color=color, label=f"{name} (AP = {ap:.3f})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# Volcano
# ---------------------------------------------------------------------------
def plot_volcano(
    df: pd.DataFrame,
    out_path: str | Path,
    logfc_col: str = "log2_fold_change",
    pval_col: str = "pvalue",
    label_col: str | None = None,
    title: str = "Volcano plot",
    n_label: int = 10,
) -> Path:
    """Volcano plot for differential TE expression / binding."""
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    df = df.copy()
    df["neg_log10_p"] = -np.log10(df[pval_col].clip(lower=1e-300))

    # Significance threshold
    sig = (df["neg_log10_p"] > -np.log10(0.05)) & (df[logfc_col].abs() > 1)
    colors = np.where(sig, NBT_PALETTE["secondary"], NBT_PALETTE["muted"])

    ax.scatter(df[logfc_col], df["neg_log10_p"], c=colors, s=10, alpha=0.5, edgecolors="none")

    if label_col and n_label > 0:
        top = df.reindex(df["neg_log10_p"].sort_values(ascending=False).index).head(n_label)
        for _, row in top.iterrows():
            ax.annotate(
                str(row[label_col]), (row[logfc_col], row["neg_log10_p"]), fontsize=7, alpha=0.8
            )

    ax.axhline(-np.log10(0.05), ls="--", color="gray", alpha=0.5)
    ax.axvline(1, ls="--", color="gray", alpha=0.5)
    ax.axvline(-1, ls="--", color="gray", alpha=0.5)
    ax.set_xlabel(logfc_col)
    ax.set_ylabel("-log10(p-value)")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = [
        Patch(facecolor=NBT_PALETTE["secondary"], label="Significant"),
        Patch(facecolor=NBT_PALETTE["muted"], label="Not significant"),
    ]
    ax.legend(handles=legend, frameon=False, loc="upper right")

    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# Heatmap with hierarchical clustering
# ---------------------------------------------------------------------------
def plot_heatmap(
    df: pd.DataFrame,
    out_path: str | Path,
    title: str = "Heatmap",
    cmap: str = "viridis",
    row_cluster: bool = True,
    col_cluster: bool = True,
) -> Path:
    """Clustered heatmap."""
    g = sns.clustermap(
        df,
        cmap=cmap,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        standard_scale=0,
        figsize=(10, 10),
        cbar_kws={"label": "Z-score"},
    )
    # Save via the clustermap figure
    fig_path = Path(out_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(fig_path, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor="white")
    plt.close("all")
    return fig_path


# ---------------------------------------------------------------------------
# PCA / UMAP
# ---------------------------------------------------------------------------
def plot_pca(
    features: np.ndarray,
    labels: np.ndarray,
    out_path: str | Path,
    label_names: dict[int, str] | None = None,
    title: str = "PCA",
) -> Path:
    """PCA scatter, coloured by label."""
    pca = PCA(n_components=2)
    xy = pca.fit_transform(features)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    scatter = ax.scatter(
        xy[:, 0], xy[:, 1], c=labels, cmap="tab10", s=8, alpha=0.5, edgecolors="none"
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(title)
    if label_names:
        legend = ax.legend(*scatter.legend_elements(), title="Class", frameon=False)
        ax.add_artist(legend)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save(fig, out_path)


def plot_umap(
    features: np.ndarray,
    labels: np.ndarray,
    out_path: str | Path,
    title: str = "UMAP",
) -> Path:
    """UMAP scatter, coloured by label."""
    import umap

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
    xy = reducer.fit_transform(features)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.scatter(
        xy[:, 0], xy[:, 1], c=labels, cmap="tab10", s=8, alpha=0.5, edgecolors="none"
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# Sankey diagram (peptide flow)
# ---------------------------------------------------------------------------
def plot_sankey(
    sources: list[str],
    targets: list[str],
    values: list[float],
    out_path: str | Path,
    title: str = "Sankey diagram",
) -> Path:
    """Sankey diagram using Plotly."""
    import plotly.graph_objects as go

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=list(set(sources + targets)), pad=15, thickness=20),
                link=dict(
                    source=[list(set(sources + targets)).index(s) for s in sources],
                    target=[list(set(sources + targets)).index(t) for t in targets],
                    value=values,
                ),
            )
        ]
    )
    fig.update_layout(title=title, font_size=10)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_path), scale=2)
    return out_path


# ---------------------------------------------------------------------------
# Network (PPI / TE-locus-peptide)
# ---------------------------------------------------------------------------
def plot_network(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    out_path: str | Path,
    title: str = "Network",
) -> Path:
    """Network plot using networkx."""
    import networkx as nx

    G = nx.from_pandas_edgelist(edges, source="source", target="target", edge_attr="weight")

    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    pos = nx.spring_layout(G, k=0.5, seed=42)
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)
    nx.draw_networkx_nodes(
        G, pos, node_size=50, node_color=NBT_PALETTE["primary"], alpha=0.6, ax=ax
    )
    if "label" in nodes.columns:
        labels = dict(zip(nodes["id"], nodes["label"], strict=False))
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=6, ax=ax)
    ax.set_title(title)
    ax.axis("off")
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------
def plot_training_curves(
    history: dict[str, list[float]],
    out_path: str | Path,
) -> Path:
    """Plot training / validation loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    axes[0].plot(history["train_loss"], label="Train", color=NBT_PALETTE["primary"], lw=2)
    axes[0].plot(history["val_loss"], label="Validation", color=NBT_PALETTE["secondary"], lw=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and validation loss")
    axes[0].legend(frameon=False)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].plot(history["lr"], color=NBT_PALETTE["tertiary"], lw=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning rate")
    axes[1].set_title("Learning rate schedule")
    axes[1].set_yscale("log")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_path: str | Path,
    title: str = "Calibration curve",
    n_bins: int = 10,
) -> Path:
    """Reliability / calibration diagram."""
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "s-", color=NBT_PALETTE["primary"], label="DeepTE-Peptide")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save(fig, out_path)
