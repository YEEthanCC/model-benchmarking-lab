"""
Visualization module for Data Lockdown benchmark results.

Generates a comprehensive set of comparative charts that evaluate model
performance across accuracy, confidence, latency, domain, difficulty, and
calibration dimensions.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Consistent styling ──────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 15,
})

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]


class DataLockdownVisualizer:
    """Creates and saves all benchmark comparison charts."""

    def __init__(self, results_df: pd.DataFrame, output_dir: Path):
        self.df = results_df
        self.output_dir = output_dir
        self.charts_dir = output_dir / "charts"
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self.models = sorted(self.df["model"].unique())
        self.colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(self.models)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_all(self):
        """Generate the full chart suite."""
        self.plot_overall_accuracy()
        self.plot_accuracy_by_domain()
        self.plot_accuracy_by_difficulty()
        self.plot_accuracy_by_regulation()
        self.plot_confidence_distribution()
        self.plot_latency_distribution()
        self.plot_confidence_vs_accuracy()
        self.plot_radar_summary()
        self.plot_dashboard()
        print(f"  Charts ({len(list(self.charts_dir.iterdir()))}) → {self.charts_dir}")

    # ------------------------------------------------------------------
    # 1. Overall accuracy bar chart
    # ------------------------------------------------------------------
    def plot_overall_accuracy(self):
        fig, ax = plt.subplots(figsize=(7, 4))
        acc = self.df.groupby("model")["is_correct"].mean().reindex(self.models)
        bars = ax.bar(self.models, acc, color=[self.colors[m] for m in self.models])
        ax.set_ylabel("Accuracy")
        ax.set_title("Overall Accuracy by Model")
        ax.set_ylim(0, 1.05)
        for bar, val in zip(bars, acc):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.1%}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", rotation=20)
        self._save(fig, "overall_accuracy")

    # ------------------------------------------------------------------
    # 2. Accuracy grouped by domain
    # ------------------------------------------------------------------
    def plot_accuracy_by_domain(self):
        pivot = self.df.groupby(["domain", "model"])["is_correct"].mean().unstack("model").reindex(columns=self.models)
        fig, ax = plt.subplots(figsize=(9, 5))
        self._grouped_bar(ax, pivot, "Accuracy by Domain", "Accuracy")
        self._save(fig, "accuracy_by_domain")

    # ------------------------------------------------------------------
    # 3. Accuracy grouped by difficulty
    # ------------------------------------------------------------------
    def plot_accuracy_by_difficulty(self):
        pivot = self.df.groupby(["difficulty", "model"])["is_correct"].mean().unstack("model").reindex(columns=self.models)
        fig, ax = plt.subplots(figsize=(9, 5))
        self._grouped_bar(ax, pivot, "Accuracy by Difficulty", "Accuracy")
        self._save(fig, "accuracy_by_difficulty")

    # ------------------------------------------------------------------
    # 4. Accuracy grouped by regulation
    # ------------------------------------------------------------------
    def plot_accuracy_by_regulation(self):
        pivot = (
            self.df.groupby(["regulation", "model"])["is_correct"]
            .mean()
            .unstack("model")
            .reindex(columns=self.models)
        )
        if pivot.empty or len(pivot) < 1:
            return
        fig, ax = plt.subplots(figsize=(max(9, len(pivot) * 1.2), 5))
        self._grouped_bar(ax, pivot, "Accuracy by Regulation", "Accuracy")
        ax.tick_params(axis="x", rotation=35)
        self._save(fig, "accuracy_by_regulation")

    # ------------------------------------------------------------------
    # 5. Confidence distribution (box plot)
    # ------------------------------------------------------------------
    def plot_confidence_distribution(self):
        fig, ax = plt.subplots(figsize=(7, 4))
        data = [self.df.loc[self.df["model"] == m, "confidence"].dropna().values for m in self.models]
        bp = ax.boxplot(data, labels=self.models, patch_artist=True, widths=0.5)
        for patch, model in zip(bp["boxes"], self.models):
            patch.set_facecolor(self.colors[model])
            patch.set_alpha(0.7)
        ax.set_ylabel("Confidence Score")
        ax.set_title("Confidence Distribution by Model")
        ax.tick_params(axis="x", rotation=20)
        self._save(fig, "confidence_distribution")

    # ------------------------------------------------------------------
    # 6. Latency distribution (box plot)
    # ------------------------------------------------------------------
    def plot_latency_distribution(self):
        fig, ax = plt.subplots(figsize=(7, 4))
        data = [self.df.loc[self.df["model"] == m, "latency"].dropna().values for m in self.models]
        bp = ax.boxplot(data, labels=self.models, patch_artist=True, widths=0.5)
        for patch, model in zip(bp["boxes"], self.models):
            patch.set_facecolor(self.colors[model])
            patch.set_alpha(0.7)
        ax.set_ylabel("Latency (s)")
        ax.set_title("Latency Distribution by Model")
        ax.tick_params(axis="x", rotation=20)
        self._save(fig, "latency_distribution")

    # ------------------------------------------------------------------
    # 7. Confidence vs accuracy (calibration scatter)
    # ------------------------------------------------------------------
    def plot_confidence_vs_accuracy(self):
        """Bin confidence into deciles and plot accuracy per bin per model."""
        fig, ax = plt.subplots(figsize=(7, 5))
        for model in self.models:
            mdf = self.df[self.df["model"] == model].copy()
            mdf["conf_bin"] = pd.cut(mdf["confidence"], bins=np.arange(0, 1.1, 0.1), include_lowest=True)
            cal = mdf.groupby("conf_bin", observed=False)["is_correct"].mean()
            midpoints = [interval.mid for interval in cal.index]
            ax.plot(midpoints, cal.values, marker="o", label=model, color=self.colors[model])

        # Perfect-calibration diagonal
        ax.plot([0, 1], [0, 1], ls="--", color="grey", alpha=0.6, label="Perfect calibration")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title("Calibration: Confidence vs Accuracy")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        self._save(fig, "confidence_vs_accuracy")

    # ------------------------------------------------------------------
    # 8. Radar / spider chart for multi-metric comparison
    # ------------------------------------------------------------------
    def plot_radar_summary(self):
        """Radar chart comparing models on accuracy, confidence, speed, and per-domain accuracy."""
        metrics: dict[str, dict[str, float]] = {}
        for model in self.models:
            mdf = self.df[self.df["model"] == model]
            metrics[model] = {
                "Accuracy": mdf["is_correct"].mean(),
                "Avg Confidence": mdf["confidence"].mean(),
                "Speed (1/latency)": 1 / max(mdf["latency"].mean(), 0.01),
            }
            # Add per-domain accuracy as extra spokes
            for domain in sorted(self.df["domain"].unique()):
                ddf = mdf[mdf["domain"] == domain]
                metrics[model][f"{domain} Acc"] = ddf["is_correct"].mean() if len(ddf) else 0

        labels = list(next(iter(metrics.values())).keys())
        n = len(labels)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]  # close the polygon

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        for model in self.models:
            vals = [metrics[model][l] for l in labels]
            # Normalise each axis to [0, 1] using max across models
            max_vals = [max(metrics[m][l] for m in self.models) or 1 for l in labels]
            normed = [v / mv for v, mv in zip(vals, max_vals)]
            normed += normed[:1]
            ax.plot(angles, normed, marker="o", label=model, color=self.colors[model])
            ax.fill(angles, normed, alpha=0.1, color=self.colors[model])

        ax.set_thetagrids(np.degrees(angles[:-1]), labels, size=8)
        ax.set_title("Multi-Metric Radar Comparison", y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        self._save(fig, "radar_summary")

    # ------------------------------------------------------------------
    # 9. Combined dashboard (2×2 overview)
    # ------------------------------------------------------------------
    def plot_dashboard(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Data Lockdown – Model Comparison Dashboard", fontsize=15, fontweight="bold")

        # Top-left: overall accuracy
        ax = axes[0, 0]
        acc = self.df.groupby("model")["is_correct"].mean().reindex(self.models)
        bars = ax.bar(self.models, acc, color=[self.colors[m] for m in self.models])
        for bar, val in zip(bars, acc):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.1%}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_title("Overall Accuracy")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=20)

        # Top-right: accuracy by domain
        ax = axes[0, 1]
        pivot = self.df.groupby(["domain", "model"])["is_correct"].mean().unstack("model").reindex(columns=self.models)
        self._grouped_bar(ax, pivot, "Accuracy by Domain", "Accuracy", show_legend=True)

        # Bottom-left: confidence box plot
        ax = axes[1, 0]
        data = [self.df.loc[self.df["model"] == m, "confidence"].dropna().values for m in self.models]
        bp = ax.boxplot(data, labels=self.models, patch_artist=True, widths=0.5)
        for patch, model in zip(bp["boxes"], self.models):
            patch.set_facecolor(self.colors[model])
            patch.set_alpha(0.7)
        ax.set_title("Confidence Distribution")
        ax.tick_params(axis="x", rotation=20)

        # Bottom-right: latency box plot
        ax = axes[1, 1]
        data = [self.df.loc[self.df["model"] == m, "latency"].dropna().values for m in self.models]
        bp = ax.boxplot(data, labels=self.models, patch_artist=True, widths=0.5)
        for patch, model in zip(bp["boxes"], self.models):
            patch.set_facecolor(self.colors[model])
            patch.set_alpha(0.7)
        ax.set_title("Latency Distribution (s)")
        ax.tick_params(axis="x", rotation=20)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        self._save(fig, "dashboard")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _grouped_bar(self, ax, pivot, title, ylabel, show_legend=True):
        """Draw a grouped bar chart from a pivoted (categories × models) DataFrame."""
        categories = pivot.index.tolist()
        n_models = len(self.models)
        x = np.arange(len(categories))
        width = 0.8 / n_models

        for i, model in enumerate(self.models):
            vals = pivot[model].values
            offset = (i - n_models / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=model, color=self.colors[model])
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                            f"{val:.0%}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, 1.15)
        if show_legend:
            ax.legend()

    def _save(self, fig, name: str):
        path = self.charts_dir / f"{name}.png"
        fig.savefig(path)
        plt.close(fig)
