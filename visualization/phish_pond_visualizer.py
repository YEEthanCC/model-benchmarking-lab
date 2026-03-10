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


class PhishPondVisualizer:
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
        # self.plot_accuracy_by_domain()
        # self.plot_accuracy_by_difficulty()
        # self.plot_accuracy_by_regulation()
        self.plot_confidence_distribution()
        self.plot_latency_distribution()
        self.plot_confidence_vs_accuracy()
        self.plot_radar_summary()
        # self.plot_dashboard()
        self.plot_per_question_answers()
        self.plot_per_question_latency()
        self.plot_per_question_confidence()
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
    # 10. Per-question answer comparison (heatmap)
    # ------------------------------------------------------------------
    def plot_per_question_answers(self):
        """Heatmap showing each model's answer vs ground truth per question.

        Cells are coloured green (correct) / red (incorrect).  The cell text
        shows the model's chosen answer letter so reviewers can spot
        systematic mis-answers at a glance.
        """
        questions_dir = self.charts_dir / "questions"
        questions_dir.mkdir(parents=True, exist_ok=True)

        qids = sorted(self.df["id"].unique())

        # Build a matrix: rows = questions, cols = models
        answer_matrix = []
        correct_matrix = []
        gt_list = []
        for qid in qids:
            qdf = self.df[self.df["id"] == qid]
            gt = qdf["ground_truth"].iloc[0]
            gt_list.append(gt)
            row_ans = []
            row_cor = []
            for model in self.models:
                mq = qdf[qdf["model"] == model]
                if mq.empty:
                    row_ans.append("")
                    row_cor.append(np.nan)
                else:
                    row_ans.append(mq["answer"].iloc[0])
                    row_cor.append(1 if mq["is_correct"].iloc[0] else 0)
            answer_matrix.append(row_ans)
            correct_matrix.append(row_cor)

        correct_arr = np.array(correct_matrix, dtype=float)

        # Paginate into chunks so individual charts stay readable
        page_size = 30
        for page_start in range(0, len(qids), page_size):
            page_end = min(page_start + page_size, len(qids))
            page_qids = qids[page_start:page_end]
            page_correct = correct_arr[page_start:page_end]
            page_answers = answer_matrix[page_start:page_end]
            page_gt = gt_list[page_start:page_end]

            n_rows = len(page_qids)
            fig_height = max(4, n_rows * 0.45 + 2)
            fig, ax = plt.subplots(figsize=(max(7, len(self.models) * 2 + 2), fig_height))

            # Custom colourmap: red (0) → green (1)
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap(["#e74c3c", "#2ecc71"])

            ax.imshow(page_correct, aspect="auto", cmap=cmap, vmin=0, vmax=1)

            # Annotate each cell with the answer letter
            for i in range(n_rows):
                for j in range(len(self.models)):
                    ax.text(j, i, page_answers[i][j],
                            ha="center", va="center", fontsize=8, fontweight="bold",
                            color="white")

            # Axis labels
            ylabels = [f"Q{qid} (GT:{gt})" for qid, gt in zip(page_qids, page_gt)]
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels(ylabels, fontsize=8)
            ax.set_xticks(range(len(self.models)))
            ax.set_xticklabels(self.models, fontsize=9, rotation=20, ha="right")
            ax.set_title(f"Per-Question Answers  (Q {page_start + 1}–{page_end})", fontsize=12, fontweight="bold")

            # Legend patches
            from matplotlib.patches import Patch
            legend_items = [Patch(facecolor="#2ecc71", label="Correct"),
                            Patch(facecolor="#e74c3c", label="Incorrect")]
            ax.legend(handles=legend_items, loc="upper right", bbox_to_anchor=(1.18, 1.02))

            fig.tight_layout()
            self._save(fig, f"questions/answer_heatmap_{page_start + 1}_{page_end}")

    # ------------------------------------------------------------------
    # 11. Per-question latency comparison (grouped bar)
    # ------------------------------------------------------------------
    def plot_per_question_latency(self):
        """Grouped bar chart of each model's response latency per question."""
        questions_dir = self.charts_dir / "questions"
        questions_dir.mkdir(parents=True, exist_ok=True)

        qids = sorted(self.df["id"].unique())

        page_size = 30
        for page_start in range(0, len(qids), page_size):
            page_end = min(page_start + page_size, len(qids))
            page_qids = qids[page_start:page_end]
            n_q = len(page_qids)

            fig_width = max(10, n_q * 0.6 + 3)
            fig, ax = plt.subplots(figsize=(fig_width, 5))

            x = np.arange(n_q)
            n_models = len(self.models)
            width = 0.8 / n_models

            for i, model in enumerate(self.models):
                latencies = []
                for qid in page_qids:
                    mq = self.df[(self.df["id"] == qid) & (self.df["model"] == model)]
                    latencies.append(mq["latency"].iloc[0] if not mq.empty else 0)
                offset = (i - n_models / 2 + 0.5) * width
                ax.bar(x + offset, latencies, width, label=model, color=self.colors[model])

            ax.set_xticks(x)
            ax.set_xticklabels([f"Q{q}" for q in page_qids], fontsize=7, rotation=45, ha="right")
            ax.set_ylabel("Latency (s)")
            ax.set_title(f"Per-Question Latency  (Q {page_start + 1}–{page_end})", fontsize=12, fontweight="bold")
            ax.legend(fontsize=8)
            fig.tight_layout()
            self._save(fig, f"questions/latency_{page_start + 1}_{page_end}")

    # ------------------------------------------------------------------
    # 12. Per-question confidence score comparison (grouped bar)
    # ------------------------------------------------------------------
    def plot_per_question_confidence(self):
        """Grouped bar chart of each model's confidence score per question,
        with markers indicating whether the answer was correct."""
        questions_dir = self.charts_dir / "questions"
        questions_dir.mkdir(parents=True, exist_ok=True)

        qids = sorted(self.df["id"].unique())

        page_size = 30
        for page_start in range(0, len(qids), page_size):
            page_end = min(page_start + page_size, len(qids))
            page_qids = qids[page_start:page_end]
            n_q = len(page_qids)

            fig_width = max(10, n_q * 0.6 + 3)
            fig, ax = plt.subplots(figsize=(fig_width, 5))

            x = np.arange(n_q)
            n_models = len(self.models)
            width = 0.8 / n_models

            for i, model in enumerate(self.models):
                confidences = []
                correctness = []
                for qid in page_qids:
                    mq = self.df[(self.df["id"] == qid) & (self.df["model"] == model)]
                    if not mq.empty:
                        confidences.append(mq["confidence"].iloc[0])
                        correctness.append(mq["is_correct"].iloc[0])
                    else:
                        confidences.append(0)
                        correctness.append(False)
                offset = (i - n_models / 2 + 0.5) * width
                bars = ax.bar(x + offset, confidences, width, label=model,
                              color=self.colors[model], alpha=0.85)
                # Place a small marker on top: ✓ for correct, ✗ for wrong
                for bar, conf, correct in zip(bars, confidences, correctness):
                    marker = "✓" if correct else "✗"
                    colour = "#2ecc71" if correct else "#e74c3c"
                    ax.text(bar.get_x() + bar.get_width() / 2, conf + 0.02,
                            marker, ha="center", va="bottom", fontsize=7,
                            fontweight="bold", color=colour)

            ax.set_xticks(x)
            ax.set_xticklabels([f"Q{q}" for q in page_qids], fontsize=7, rotation=45, ha="right")
            ax.set_ylabel("Confidence Score")
            ax.set_ylim(0, 1.15)
            ax.set_title(f"Per-Question Confidence  (Q {page_start + 1}–{page_end})", fontsize=12, fontweight="bold")
            ax.legend(fontsize=8)
            fig.tight_layout()
            self._save(fig, f"questions/confidence_{page_start + 1}_{page_end}")

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

