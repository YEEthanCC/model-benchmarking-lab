"""
Benchmarking pipeline for Data Lockdown question sets.
Runs pluggable models over security/privacy questions, stores structured
results, and generates comparative visualizations.
"""

import json
import time
from datetime import datetime
from typing import Optional
from pathlib import Path

import pandas as pd
from azure.ai.projects import AIProjectClient
from tqdm import tqdm

from agents.azure_agent import AzureAgent, AgentResponse
from agents.prompts.data_lockdown_prompts import SYSTEM_PROMPT, USER_PROMPT
from preprocessing.data_lockdown_preprocessor import DataLockdownPreprocessor
from visualization.data_lockdown_visualizer import DataLockdownVisualizer

RESULTS_DIR = Path("results")
ANSWER_MAP = {"A": "option_A", "B": "option_B", "C": "option_C", "D": "option_D"}


def get_agent_response(row, agent: AzureAgent) -> AgentResponse:
    options = f"""
a: {row["option_A"]}
b: {row["option_B"]}
c: {row["option_C"]}
d: {row["option_D"]}
"""
    prompt = USER_PROMPT.format(question=row["question"], options=options)
    return agent.run(prompt)


class DataLockdownPipeline:
    """Orchestrates multi-model benchmarking over Data Lockdown questions."""

    def __init__(
        self,
        client: AIProjectClient,
        models: list[str],
        file_path: str,
    ):
        self.client = client
        self.models = models
        self.file_path = file_path

        # Initialize one agent per model
        self.agents = []
        for model in models:
            self.agents.append(AzureAgent(client, model, SYSTEM_PROMPT, None))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def execute(self):
        """Run all models, persist results, and generate visualizations."""
        try:
            processor = DataLockdownPreprocessor(self.file_path)
            df = processor.process()
        except Exception as e:
            print(f"Invalid data format: {e}")
            return

        tqdm.pandas()

        # Collect per-question results for every model
        all_results: list[pd.DataFrame] = []

        for index, agent in enumerate(self.agents):
            model_name = self.models[index]
            print(f"\n--- Running {model_name} ---")
            responses = df.progress_apply(get_agent_response, axis=1, agent=agent)
            print(f"{model_name} completed")

            model_df = self._build_model_results(df, responses, model_name)
            all_results.append(model_df)

        # Merge all models into one results DataFrame
        results_df = pd.concat(all_results, ignore_index=True)

        # Persist to disk
        run_dir = self._save_results(results_df)

        # Generate visualizations
        self._generate_visualizations(results_df, run_dir)

        print(f"\nResults and charts saved to {run_dir}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_model_results(
        questions_df: pd.DataFrame,
        responses: pd.Series,
        model_name: str,
    ) -> pd.DataFrame:
        """Combine question metadata with agent responses into a flat DataFrame."""
        records = []
        for (_, row), resp in zip(questions_df.iterrows(), responses):
            is_correct = resp.answer == row["ground_truth"]
            records.append(
                {
                    "id": row["id"],
                    "domain": row["domain"],
                    "difficulty": row["difficulty"],
                    "regulation": row["regulation"],
                    "question": row["question"],
                    "ground_truth": row["ground_truth"],
                    "model": model_name,
                    "answer": resp.answer,
                    "is_correct": is_correct,
                    "confidence": resp.confidence,
                    "reasoning": resp.reasoning,
                    "latency": resp.latency,
                }
            )
        return pd.DataFrame(records)

    def _save_results(self, results_df: pd.DataFrame) -> Path:
        """Write per-question CSV and aggregate summary JSON to a timestamped folder."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RESULTS_DIR / f"data_lockdown_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # ---------- Per-question detail ----------
        detail_path = run_dir / "detailed_results.csv"
        results_df.to_csv(detail_path, index=False)
        print(f"  Detailed results → {detail_path}")

        # ---------- Per-model summary ----------
        summary = self._compute_summary(results_df)
        summary_path = run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary           → {summary_path}")

        return run_dir

    @staticmethod
    def _compute_summary(df: pd.DataFrame) -> dict:
        """Compute aggregate metrics per model."""
        summary: dict = {"generated_at": datetime.now().isoformat(), "models": {}}
        for model, grp in df.groupby("model"):
            accuracy = grp["is_correct"].mean()
            model_summary = {
                "total_questions": len(grp),
                "correct": int(grp["is_correct"].sum()),
                "accuracy": round(accuracy, 4),
                "avg_confidence": round(grp["confidence"].mean(), 4),
                "avg_latency_s": round(grp["latency"].mean(), 4),
                "median_latency_s": round(grp["latency"].median(), 4),
                "by_domain": {},
                "by_difficulty": {},
                "by_regulation": {},
            }
            for domain, dgrp in grp.groupby("domain"):
                model_summary["by_domain"][domain] = {
                    "accuracy": round(dgrp["is_correct"].mean(), 4),
                    "count": len(dgrp),
                    "avg_confidence": round(dgrp["confidence"].mean(), 4),
                }
            for diff, dgrp in grp.groupby("difficulty"):
                model_summary["by_difficulty"][str(diff)] = {
                    "accuracy": round(dgrp["is_correct"].mean(), 4),
                    "count": len(dgrp),
                    "avg_confidence": round(dgrp["confidence"].mean(), 4),
                }
            for reg, dgrp in grp.groupby("regulation"):
                model_summary["by_regulation"][reg] = {
                    "accuracy": round(dgrp["is_correct"].mean(), 4),
                    "count": len(dgrp),
                }
            summary["models"][model] = model_summary
        return summary

    @staticmethod
    def _generate_visualizations(results_df: pd.DataFrame, run_dir: Path):
        """Create all comparative charts and save into the run directory."""
        viz = DataLockdownVisualizer(results_df, run_dir)
        viz.generate_all()

