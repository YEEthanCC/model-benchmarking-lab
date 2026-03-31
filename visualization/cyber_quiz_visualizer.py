"""
Visualization module for Data Lockdown benchmark results.

Generates a comprehensive set of comparative charts that evaluate model
performance across accuracy, confidence, latency, domain, difficulty, and
calibration dimensions.
"""

import warnings
from pathlib import Path

import matplotlib
from matplotlib.patches import Patch
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


class CyberQuizVisualizer:
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
        self.plot_per_question_answers()
        self.plot_per_question_latency()
        # self.plot_gpt_per_question()
        print(f"  Charts ({len(list(self.charts_dir.iterdir()))}) → {self.charts_dir}")

    # ------------------------------------------------------------------
    # 10. Per-question answer distribution (one chart per question)
    # ------------------------------------------------------------------
    def plot_per_question_answers(self):
        """Bar chart per question showing how many models chose each answer."""
        answer_dir = self.charts_dir / "questions" / "answers"
        answer_dir.mkdir(parents=True, exist_ok=True)

        for qid, grp in self.df.groupby("qid"):
            ground_truth = grp["ground_truth"].iloc[0]
            question_type = grp["question_type"].iloc[0]
            display_q = "AI Answer Distribution"
            legend_items = list[Patch]
            if str(question_type) == "TrueFalse":
                if ground_truth == 'TRUE':
                    legend_items = [
                        Patch(facecolor="#2ecc71", edgecolor="black", label="TRUE"),
                        Patch(facecolor="#e74c3c", edgecolor="black", label="FALSE"),
                    ]
                else:
                    legend_items = [
                        Patch(facecolor="#2ecc71", edgecolor="black", label="FALSE"),
                        Patch(facecolor="#e74c3c", edgecolor="black", label="TRUE"),
                    ]
                option_models: dict[str, list[str]] = {"TRUE": [], "FALSE": []}
                for _, row in grp.iterrows():
                    ans = str(row["answer"])
                    option_models.setdefault(ans, []).append(row["model"])

                options = ["TRUE", "FALSE"]
                full_options = options
                counts = [len(option_models.get(o, [])) for o in options]
                bar_colors = ["#2ecc71" if o == str(ground_truth) else "#e74c3c" for o in options]
                x = np.array([0, 0.5]) 
            elif str(question_type) == "MultipleChoice" or str(question_type) == "ImageMultipleChoice":
                legend_items = [
                    Patch(facecolor="#2ecc71", edgecolor="black", label="Correct"),
                    Patch(facecolor="#e74c3c", edgecolor="black", label="Incorrect"),
                ]
                full_options = [str(grp["a"].iloc[0]), str(grp["b"].iloc[0]), str(grp["c"].iloc[0]), str(grp["d"].iloc[0])]
                option_models: dict[str, list[str]] = {o: [] for o in full_options}
                for _, row in grp.iterrows():
                    ans = str(row["normalized_answer"])
                    option_models.setdefault(ans, []).append(row["model"])

                counts = [len(option_models.get(o, [])) for o in full_options]
                bar_colors = ["#2ecc71" if o == str(ground_truth) else "#e74c3c" for o in full_options]
                truncate = lambda s: (s[:17] + "...") if len(s) > 20 else s
                options = [truncate(o) for o in full_options]
                x = np.array([0, 0.5, 1, 1.5])
            else:
                legend_items = [
                    Patch(facecolor="#2ecc71", edgecolor="black", label="Correct"),
                    Patch(facecolor="#e74c3c", edgecolor="black", label="Incorrect"),
                ]
                option_models: dict[str, list[str]] = {"Correct": [], "Incorrect": []}
                for _, row in grp.iterrows():
                    ans = str(row["answer"])
                    if ans == ground_truth:
                        option_models.setdefault("Correct", []).append(row["model"])
                    else:
                        option_models.setdefault("Incorrect", []).append(row["model"])

                options = ["Correct", "Incorrect"]
                full_options = options
                counts = [len(option_models.get(o, [])) for o in options]
                bar_colors = ["#2ecc71" if o == "Correct" else "#e74c3c" for o in options]
                x = np.array([0, 0.5]) 

            # ── Black background ──────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(max(6, len(options) * 2), 5), facecolor="black")
            ax.set_facecolor("black")
            # ─────────────────────────────────────────────────────────────────


            bars = ax.bar(x, counts, width=0.1, color=bar_colors, edgecolor="black", linewidth=0.5)

            # Annotate model names above each bar
            for i, (bar, opt) in enumerate(zip(bars, full_options)):
                models_text = "\n".join(option_models[opt])
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.15,
                    models_text,
                    ha="center", va="bottom", fontsize=7, fontweight="bold",
                    color="white",                                    # ← white
                )

            ax.set_xticks(x)
            ax.set_xticklabels(options, fontsize=9, color="white")  # ← white        # ← white
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.tick_params(colors="white")                            # ← tick marks white
            ax.set_xlim(-0.5, len(x) / 2)
            ax.set_ylim(0, max(counts) + max(1, len(self.models) * 0.35))
            # ax.set_title(
            #     f"Q{qid}: {display_q}\n(Ground Truth: {ground_truth})",
            #     fontsize=10, color="white",                           # ← white
            # )
            ax.set_title(display_q, fontsize=13, fontweight='bold', loc='left',
                        pad=10, color="white")    
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color("#444444")                    # ← subtle on black
            ax.spines['bottom'].set_color("#444444")                  # ← subtle on black

            legend = ax.legend(
                handles=legend_items,
                fontsize=9,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.2),
                ncol=2,                        # hardcode 2 since there are always 2 answers
                framealpha=0.3,
                frameon=False
            )
            for text in legend.get_texts():
                text.set_color("white")                               # ← legend text white

            fig.tight_layout()
            self._save(fig, f"questions/answers/question_{qid}")


    # ------------------------------------------------------------------
    # 11. Per-question latency comparison (one chart per question)
    # ------------------------------------------------------------------
    def plot_per_question_latency(self, bin_seconds: int = 5):
        """Bar chart per question showing each model's response latency."""
        latency_dir = self.charts_dir / "questions" / "latency"
        latency_dir.mkdir(parents=True, exist_ok=True)

        custom_bin_edges = np.array([0, 5, 10, 15, 20, 30, 45, 60, 120, np.inf])
        custom_bin_labels = [
            "0-5s", "5-10s", "10-15s", "15-20s",
            "20-30s", "30-45s", "45-60s", "60-120s", ">120s"
        ]

        global_bins = pd.IntervalIndex.from_breaks(
            [0, 5, 10, 15, 20, 30, 45, 60, 120, float("inf")], closed="left"
        )

        max_models_per_question = int(self.df.groupby("qid")["model"].nunique().max())
        y_upper = max(1, max_models_per_question) + 1

        for qid, grp in self.df.groupby("qid"):
            title = f"AI Answer Distribution By Response Time"
            df = grp.copy()
            df['bin'] = pd.cut(
                df['latency'],
                bins=custom_bin_edges,
                right=False,
                include_lowest=True
            )

            answers = df['answer'].unique()
            pivot = (
                df.groupby(['bin', 'answer'], observed=True)
                .size()
                .unstack(fill_value=0)
                .reindex(global_bins, fill_value=0)
            )
            for a in answers:
                if a not in pivot.columns:
                    pivot[a] = 0

            # models_by_bin = (
            #     df.groupby('bin', observed=True)['model']
            #     .apply(lambda s: ", ".join(str(m).split('-')[0].upper() for m in s.tolist()))
            #     .reindex(global_bins, fill_value="")
            # )
            models_by_bin = (
                df.groupby('bin', observed=True).apply(
                    lambda group: "\n".join(
                        f"{str(row['model']).split('-')[0].upper()} ({row['confidence']:.2f})"
                        for _, row in group.iterrows()
                    )
                ).reindex(global_bins, fill_value="")
            )

            x = np.arange(len(custom_bin_labels))
            bar_width = 0.7

            # ── Black background ──────────────────────────────────────────────
            fig, ax = plt.subplots(
                figsize=(max(6, len(custom_bin_labels) * 0.8 + 1), 4.5),
                facecolor="black"
            )
            ax.set_facecolor("black")
            # ─────────────────────────────────────────────────────────────────

            bottom = np.zeros(len(x))
            for i, answer in enumerate(answers):
                vals = pivot[answer].values.astype(float)
                color = "#2ecc71" if answer == df["ground_truth"].iloc[0] else "#e74c3c"
                label = answer[:20] + "..." if len(answer) > 20 else answer
                ax.bar(x, vals, bar_width, bottom=bottom, color=color, label=label,
                    edgecolor="black", linewidth=0.5, zorder=2)   # ← edgecolor black
                bottom += vals

            ax.set_xticks(x)
            ax.set_xticklabels(custom_bin_labels, fontsize=10, rotation=45, ha='right',
                            color="white")                         # ← tick labels white
            # ax.set_ylabel("Number of Models", fontsize=11, color="white")   # ← white
            ax.set_xlabel("Response Time", fontsize=11, color="white")      # ← white
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.tick_params(colors="white")                            # ← tick marks white
            ax.set_xlim(-0.5, len(x) - 0.5)
            ax.set_ylim(0, y_upper)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color("#444444")                    # ← subtle on black
            ax.spines['bottom'].set_color("#444444")                  # ← subtle on black

            ax.set_title(title, fontsize=13, fontweight='bold', loc='left',
                        pad=10, color="white")                       # ← white

            value_to_label = {
                "phishing": "Scam",
                "real": "Legitimate",
            }

            # Update the legend labels using the mapping
            mapped_labels = [value_to_label.get(answer, answer) for answer in answers]
            # if df["ground_truth"].iloc[0] == 'phishing':
            #     legend_items = [
            #         Patch(facecolor="#2ecc71", edgecolor="black", label="Scam"),
            #         Patch(facecolor="#e74c3c", edgecolor="black", label="Legitimate"),
            #     ]
            # else:
            #     legend_items = [
            #         Patch(facecolor="#2ecc71", edgecolor="black", label="Legitimate"),
            #         Patch(facecolor="#e74c3c", edgecolor="black", label="Scam"),
            #     ]
            legend_items = [
                Patch(facecolor="#2ecc71", edgecolor="black", label="Correct"),
                Patch(facecolor="#e74c3c", edgecolor="black", label="Incorrect"),
            ]
            legend = ax.legend(
                handles=legend_items,
                fontsize=9,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.45),
                ncol=2,                        # hardcode 2 since there are always 2 answers
                framealpha=0.3,
                frameon=False
            )
            fig.subplots_adjust(bottom=0.35)  # Increase bottom margin to prevent overlap
            for text in legend.get_texts():
                text.set_color("white")                               # ← legend text white


            for xi, total, model_label in zip(x, bottom, models_by_bin.values):
                if total <= 0 or not model_label:
                    continue
                ax.text(xi, total + 0.05, model_label,
                        ha='center', va='bottom', fontsize=8, rotation=0,
                        zorder=3, color="white")                      # ← white

            fig.tight_layout()
            self._save(fig, f"questions/latency/question_{qid}")


    # ------------------------------------------------------------------
    # 14. GPT-only per-question gauge (half-doughnut speedometer)
    # ------------------------------------------------------------------
    def plot_gpt_per_question(self):
        from matplotlib.patches import Wedge

        gpt_model = next((m for m in self.models if m.lower().startswith("gpt")), None)
        if gpt_model is None:
            return

        gpt_dir = self.charts_dir / "questions" / "gpt"
        gpt_dir.mkdir(parents=True, exist_ok=True)

        gpt_df = self.df[self.df["model"] == gpt_model]

        for qid, grp in gpt_df.groupby("qid"):
            row = grp.iloc[0]
            display_q = "GPT's Performance"
            ground_truth = row["ground_truth"]
            answer = str(row["answer"])
            correct = bool(row["is_correct"])
            confidence = float(row["confidence"])
            latency = float(row["latency"])

            fill_color = "#2ecc71" if correct else "#e74c3c"
            bg_color = "#333333"                          # dark arc background
            label = "Correct" if correct else "Incorrect"

            fill_angle = confidence * 180.0

            # ── Black background ──────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(6, 4), facecolor="black")
            ax.set_facecolor("black")
            # ─────────────────────────────────────────────────────────────────

            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-0.4, 1.4)
            ax.set_aspect("equal")
            ax.axis("off")

            # Background semicircle
            bg_wedge = Wedge((0, 0), 1.0, 0, 180, width=0.35, facecolor=bg_color,
                            edgecolor="black", lw=2)
            ax.add_patch(bg_wedge)

            # Filled portion
            if confidence > 0:
                fill_wedge = Wedge((0, 0), 1.0, 180 - fill_angle, 180, width=0.35,
                                facecolor=fill_color, edgecolor="black", lw=2)
                ax.add_patch(fill_wedge)

            # "Correct" / "Incorrect" at the top
            ax.text(0, 1.25, label, ha="center", va="center",
                    fontsize=14, fontweight="bold", color=fill_color)

            # Latency in the centre — white instead of dark slate
            ax.text(0, 0.35, f"{latency:.2f}s", ha="center", va="center",
                    fontsize=16, fontweight="bold", color="white")       # ← changed
            ax.text(0, 0.18, "latency", ha="center", va="center",
                    fontsize=9, color="#aaaaaa")                         # ← changed

            # Confidence label — white instead of dark slate
            ax.text(0, -0.05, f"Confidence: {confidence:.0%}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white")       # ← changed

            # Title — white
            fig.suptitle(
                f"Q{qid}\n{display_q}\n"
                f"Answer: {answer}  |  Ground Truth: {ground_truth}",
                fontsize=9, fontweight="bold", y=0.98, color="white",   # ← changed
            )
            fig.tight_layout(rect=[0, 0, 1, 0.82])
            self._save(fig, f"questions/gpt/question_{qid}")



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

