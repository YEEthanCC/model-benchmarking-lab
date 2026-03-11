"""
Visualization module for Data Lockdown benchmark results.

Generates a comprehensive set of comparative charts that evaluate model
performance across accuracy, confidence, latency, domain, difficulty, and
calibration dimensions.
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

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
        self.plot_per_question_heatmap()
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
    # 10. Per-question answer distribution (one chart per question)
    # ------------------------------------------------------------------
    def plot_per_question_answers(self):
        """Bar chart per question showing how many models chose each answer.

        The correct-answer bar is green; others are grey.  Model names that
        picked each option are listed above the corresponding bar.
        """
        answer_dir = self.charts_dir / "questions" / "answers"
        answer_dir.mkdir(parents=True, exist_ok=True)

        for qid, grp in self.df.groupby("id"):
            ground_truth = grp["ground_truth"].iloc[0]
            question_text = str(grp["question"].iloc[0])
            display_q = question_text if len(question_text) <= 120 else question_text[:117] + "..."

            # Collect which models chose which answer
            option_models: dict[str, list[str]] = {"phishing": [], "real": []}
            for _, row in grp.iterrows():
                ans = str(row["answer"])
                option_models.setdefault(ans, []).append(row["model"])

            options = ["phishing", "real"]
            counts = [len(option_models.get(o, [])) for o in options]
            bar_colors = ["#2ecc71" if o == str(ground_truth) else "#95a5a6" for o in options]

            fig, ax = plt.subplots(figsize=(max(6, len(options) * 2), 5))
            x = np.arange(len(options))
            bars = ax.bar(x, counts, color=bar_colors, edgecolor="black", linewidth=0.5)

            # Annotate model names above each bar
            for i, (bar, opt) in enumerate(zip(bars, options)):
                models_text = "\n".join(option_models[opt])
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.15,
                    models_text,
                    ha="center", va="bottom", fontsize=7, fontweight="bold",
                )

            ax.set_xticks(x)
            ax.set_xticklabels(options, fontsize=9)
            ax.set_ylabel("Number of Models")
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_ylim(0, max(counts) + max(1, len(self.models) * 0.35))
            ax.set_title(
                f"Q{qid}: {display_q}\n(Ground Truth: {ground_truth})",
                fontsize=10,
            )

            from matplotlib.patches import Patch
            legend_items = [
                Patch(facecolor="#2ecc71", edgecolor="black", label="Correct option"),
                Patch(facecolor="#95a5a6", edgecolor="black", label="Incorrect option"),
            ]
            ax.legend(handles=legend_items, loc="best", fontsize=8)

            fig.tight_layout()
            self._save(fig, f"questions/answers/question_{qid}")

    # ------------------------------------------------------------------
    # 11. Per-question latency comparison (one chart per question)
    # ------------------------------------------------------------------
    def plot_per_question_latency(self):
        """Bar chart per question showing each model's response latency."""
        latency_dir = self.charts_dir / "questions" / "latency"
        latency_dir.mkdir(parents=True, exist_ok=True)

        for qid, grp in self.df.groupby("id"):
            question_text = str(grp["question"].iloc[0])
            display_q = question_text if len(question_text) <= 120 else question_text[:117] + "..."

            models_present = [m for m in self.models if m in grp["model"].values]
            if not models_present:
                continue

            latencies = []
            for m in models_present:
                mrow = grp[grp["model"] == m]
                latencies.append(float(mrow["latency"].iloc[0]) if not mrow.empty else 0)

            fig, ax = plt.subplots(figsize=(max(6, len(models_present) * 1.8), 4))
            x = np.arange(len(models_present))
            colors = [self.colors.get(m, "#999999") for m in models_present]
            bars = ax.bar(x, latencies, color=colors, edgecolor="black", linewidth=0.5, width=0.5)

            for bar, val in zip(bars, latencies):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                        f"{val:.2f}s", ha="center", va="bottom", fontsize=8, fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(models_present, fontsize=9, rotation=20)
            ax.set_ylabel("Latency (s)")
            ax.set_title(f"Q{qid}: {display_q}", fontsize=10)
            fig.tight_layout()
            self._save(fig, f"questions/latency/question_{qid}")

    # ------------------------------------------------------------------
    # 12. Per-question confidence score comparison (one chart per question)
    # ------------------------------------------------------------------
    def plot_per_question_confidence(self):
        """Bar chart per question showing each model's confidence score,
        with correctness markers above each bar."""
        conf_dir = self.charts_dir / "questions" / "confidence"
        conf_dir.mkdir(parents=True, exist_ok=True)

        for qid, grp in self.df.groupby("id"):
            question_text = str(grp["question"].iloc[0])
            display_q = question_text if len(question_text) <= 120 else question_text[:117] + "..."

            models_present = [m for m in self.models if m in grp["model"].values]
            if not models_present:
                continue

            confidences = []
            correctness = []
            for m in models_present:
                mrow = grp[grp["model"] == m]
                if not mrow.empty:
                    confidences.append(float(mrow["confidence"].iloc[0]))
                    correctness.append(bool(mrow["is_correct"].iloc[0]))
                else:
                    confidences.append(0)
                    correctness.append(False)

            fig, ax = plt.subplots(figsize=(max(6, len(models_present) * 1.8), 4))
            x = np.arange(len(models_present))
            colors = [self.colors.get(m, "#999999") for m in models_present]
            bars = ax.bar(x, confidences, color=colors, edgecolor="black",
                          linewidth=0.5, width=0.5, alpha=0.85)

            for bar, conf, correct in zip(bars, confidences, correctness):
                marker = "✓" if correct else "✗"
                colour = "#2ecc71" if correct else "#e74c3c"
                ax.text(bar.get_x() + bar.get_width() / 2, conf + 0.03,
                        marker, ha="center", va="bottom", fontsize=11,
                        fontweight="bold", color=colour)

            ax.set_xticks(x)
            ax.set_xticklabels(models_present, fontsize=9, rotation=20)
            ax.set_ylabel("Confidence Score")
            ax.set_ylim(0, 1.15)
            ax.set_title(f"Q{qid}: {display_q}", fontsize=10)
            fig.tight_layout()
            self._save(fig, f"questions/confidence/question_{qid}")

    # ------------------------------------------------------------------
    # 13. Per-question heatmap (answer / confidence / latency)
    # ------------------------------------------------------------------
    def plot_per_question_heatmap(self):
        """One heatmap per question.  Columns: Answer, Confidence, Latency.
        Rows: models.  Cell colours encode correctness / performance.

        * Answer   – green if correct, red if wrong.
        * Confidence – green (≥0.6) / red (<0.6), darker at the extremes.
        * Latency  – same colour logic applied to  1 / latency  (clamped to [0,1]).
        """
        heatmap_dir = self.charts_dir / "questions" / "heatmaps"
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        for qid, grp in self.df.groupby("id"):
            question_text = str(grp["question"].iloc[0])
            display_q = (
                question_text if len(question_text) <= 120
                else question_text[:117] + "..."
            )
            ground_truth = grp["ground_truth"].iloc[0]

            models_present = [m for m in self.models if m in grp["model"].values]
            if not models_present:
                continue

            n_models = len(models_present)
            cols = ["Answer", "Confidence", "Latency"]

            cell_colors: list[list[str]] = []
            cell_texts: list[list[str]] = []

            for m in models_present:
                mrow = grp[grp["model"] == m].iloc[0]
                correct = bool(mrow["is_correct"])
                answer = "Correct" if correct else "Incorrect"
                confidence = float(mrow["confidence"])
                latency = float(mrow["latency"])
                speed = min(1.0, latency / 40)

                ans_color = "#2ecc71" if correct else "#e74c3c"
                conf_color = self._metric_confidence_color(confidence)
                lat_color = self._metric_latency_color(speed)

                cell_colors.append([ans_color, conf_color, lat_color])
                cell_texts.append([answer, f"{confidence:.2f}", f"{latency:.2f}s"])

            fig_height = max(2.5, n_models * 0.7 + 1.5)
            fig, ax = plt.subplots(figsize=(7, fig_height))
            ax.set_xlim(0, 3)
            ax.set_ylim(n_models, -1)
            ax.axis("off")

            # Header row
            for j, col in enumerate(cols):
                ax.add_patch(plt.Rectangle(
                    (j, -1), 1, 1, facecolor="#2c3e50",
                    edgecolor="white", lw=1.5,
                ))
                ax.text(j + 0.5, -0.5, col, ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white")

            # Data cells
            for i in range(n_models):
                for j in range(3):
                    ax.add_patch(plt.Rectangle(
                        (j, i), 1, 1, facecolor=cell_colors[i][j],
                        edgecolor="white", lw=1.5,
                    ))
                    ax.text(j + 0.5, i + 0.5, cell_texts[i][j],
                            ha="center", va="center", fontsize=9,
                            fontweight="bold", color="white")

            # Model labels on the left
            for i, model in enumerate(models_present):
                ax.text(-0.08, i + 0.5, model.split('-')[0].upper(), ha="right", va="center",
                        fontsize=9)

            ax.set_title(
                f"Q{qid}: {display_q}\n(Ground Truth: {ground_truth})",
                fontsize=10, pad=12,
            )
            fig.tight_layout()
            self._save(fig, f"questions/heatmaps/question_{qid}")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _metric_confidence_color(value: float) -> str:
        """Map a 0–1 metric value to a colour.

        ≥ 0.6 → green (lighter near 0.6, much darker toward 1.0)
        < 0.6 → red   (lighter near 0.6, much darker toward 0.0)
        """
        value = max(0.0, min(1.0, value))
        if value >= 0.7:
            # colour = "#2ecc71" if correct else "#e74c3c"
            t = (1 - value) / 0.3         # pale green (#e8fce8) → deep green (#033603)
            r = int(0x2E - (0x2E - 0x03) * t)
            g = int(0xCC - (0xCC - 0x36) * t)
            b = int(0x71 - (0x71 - 0x03) * t)
        else:
            t = (0.7 - value) / 0.7 # pale red (#fce8e8) → deep red (#500303)
            r = int(0xE7 - (0xE7 - 0x50) * t)
            g = int(0x4C - (0x4C - 0x03) * t)
            b = int(0x3C - (0x3C - 0x03) * t)
        return f"#{r:02x}{g:02x}{b:02x}"
    
        # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _metric_latency_color(value: float) -> str:
        """Map a 0–1 metric value to a colour.

        ≥ 0.6 → green (lighter near 0.6, much darker toward 1.0)
        < 0.6 → red   (lighter near 0.6, much darker toward 0.0)
        """
        value = max(0.0, min(1.0, value))
        if value <= 0.3:
            # colour = "#2ecc71" if correct else "#e74c3c"
            t = value / 0.3             # pale green (#e8fce8) → deep green (#033603)
            r = int(0x2E - (0x2E - 0x03) * t)
            g = int(0xCC - (0xCC - 0x36) * t)
            b = int(0x71 - (0x71 - 0x03) * t)
        else:
            t = (value - 0.3 )  / 0.7       # pale red (#fce8e8) → deep red (#500303)
            r = int(0xE7 - (0xE7 - 0x50) * t)
            g = int(0x4C - (0x4C - 0x03) * t)
            b = int(0x3C - (0x3C - 0x03) * t)
        return f"#{r:02x}{g:02x}{b:02x}"

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

