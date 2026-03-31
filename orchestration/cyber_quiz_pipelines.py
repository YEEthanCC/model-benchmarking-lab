"""
Benchmarking pipeline for Data Lockdown question sets.
Runs pluggable models over security/privacy questions, stores structured
results, and generates comparative visualizations.
"""

import json
import re
import time
from datetime import datetime
from typing import Optional
from pathlib import Path

import pandas as pd
from azure.ai.projects import AIProjectClient
from tqdm import tqdm

from agents.azure_agent import AzureAgent, AgentResponse
from agents.prompts.cyber_quiz_prompts import (
    SYSTEM_PROMPT,
    TRUE_FALSE_PROMPT,
    RANKING_PROMPT,
    MULTIPLECHOICE_PROMPT,
    MULTISELECT_CLASSIFICATION_PROMPT,
)
from preprocessing.cyber_quiz_preprocessor import CyberQuizPreprocessor
from visualization.data_lockdown_visualizer import DataLockdownVisualizer
from visualization.cyber_quiz_visualizer import CyberQuizVisualizer

RESULTS_DIR = Path("results")
ANSWER_MAP = {"A": "option_A", "B": "option_B", "C": "option_C", "D": "option_D"}


def _is_populated(value) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip() != ""


def _extract_non_empty_options(row) -> list[str]:
    option_keys = ["a", "b", "c", "d", "e", "f"]
    return [str(row[key]).strip() for key in option_keys if key in row and _is_populated(row[key])]


def _extract_classification_items(row) -> list[str]:
    # In this CSV format, classification items are directly provided via OptionA..OptionF.
    return _extract_non_empty_options(row)


def _extract_allowed_labels(row) -> list[str]:
    question = str(row.get("question", ""))

    # Example pattern: "Classify each item as PII or Non-PII."
    match = re.search(r"\bas\s+(.+?)\s+or\s+(.+?)(?:[\.?]|$)", question, flags=re.IGNORECASE)
    if match:
        first = match.group(1).strip().strip('"')
        second = match.group(2).strip().strip('"')
        if first and second:
            return [first, second]

    # Fallback: derive labels from ground truth mappings.
    gt = str(row.get("ground_truth", ""))
    labels: list[str] = []
    for segment in [part.strip() for part in gt.split(";") if part.strip()]:
        if "→" in segment:
            _, label = [piece.strip() for piece in segment.split("→", 1)]
        elif "->" in segment:
            _, label = [piece.strip() for piece in segment.split("->", 1)]
        else:
            continue
        if label and label not in labels:
            labels.append(label)
    return labels[:2]


def _normalize_spaces(value: str) -> str:
    return " ".join(str(value).strip().split())


def _parse_classification_pairs(value: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for segment in [part.strip() for part in str(value).split(";") if part.strip()]:
        if ":" in segment:
            left, right = [piece.strip() for piece in segment.split(":", 1)]
        else:
            return []
        if not left or not right:
            return []
        pairs.append((_normalize_spaces(left), _normalize_spaces(right)))
    return pairs


def _normalize_answer_for_scoring(row, answer: str) -> str:
    question_type = str(row.get("type", ""))
    answer = str(answer).strip()

    if question_type == "TrueFalse":
        upper = answer.upper()
        if upper in {"TRUE", "FALSE"}:
            return upper
        return _normalize_spaces(answer)

    if question_type == "MultipleChoice":
        letter = answer.lower()
        if letter in {"a", "b", "c", "d"}:
            mapped = row.get(f"option_{letter.upper()}")
            if _is_populated(mapped):
                return _normalize_spaces(str(mapped))
        return _normalize_spaces(answer)

    if question_type == "Ranking":
        steps = _parse_ranked_answer(answer)
        if not steps:
            return _normalize_spaces(answer)
        return "; ".join(f"{idx}. {_normalize_spaces(step)}" for idx, step in enumerate(steps, start=1))

    if question_type == "MultiSelectClassification":
        pairs = _parse_classification_pairs(answer)
        if not pairs:
            return _normalize_spaces(answer)
        return "; ".join(f"{item} → {label}" for item, label in pairs)

    return _normalize_spaces(answer)


def _is_correct_answer(row, answer: str) -> bool:
    normalized_answer = _normalize_answer_for_scoring(row, answer)
    normalized_truth = _normalize_answer_for_scoring(row, row.get("ground_truth", ""))
    return normalized_answer == normalized_truth


def _parse_ranked_answer(answer: str) -> list[str]:
    parts = [segment.strip() for segment in str(answer).split(";") if segment.strip()]
    parsed: list[str] = []
    for idx, part in enumerate(parts, start=1):
        match = re.fullmatch(rf"{idx}\.\s+(.+)", part)
        if not match:
            return []
        parsed.append(match.group(1).strip())
    return parsed


def _is_valid_answer_format(row, answer: str) -> bool:
    question_type = row["type"]
    if question_type == "TrueFalse":
        return str(answer).strip().upper() in {"TRUE", "FALSE"}

    if question_type == "Ranking":
        steps = _extract_non_empty_options(row)
        ranked_steps = [_normalize_spaces(step) for step in _parse_ranked_answer(answer)]
        expected_steps = [_normalize_spaces(step) for step in steps]
        return len(ranked_steps) == len(expected_steps) and set(ranked_steps) == set(expected_steps)

    if question_type == "MultiSelectClassification":
        items = [_normalize_spaces(item) for item in _extract_classification_items(row)]
        labels = [_normalize_spaces(label) for label in _extract_allowed_labels(row)]

        parsed_pairs = _parse_classification_pairs(answer)
        if len(parsed_pairs) != len(items):
            return False

        parsed_items: list[str] = []
        for left, right in parsed_pairs:
            if right not in labels:
                return False
            parsed_items.append(left)
        return parsed_items == items

    return str(answer).strip().lower() in {"a", "b", "c", "d", "e", "f"}


def get_agent_response(row, agent: AzureAgent) -> AgentResponse:
    if row["type"] == "TrueFalse":
        prompt = TRUE_FALSE_PROMPT.format(question=row["question"])
    elif row["type"] == "Ranking":
        steps = _extract_non_empty_options(row)
        options = "\n".join(f"{idx}. {step}" for idx, step in enumerate(steps, start=1))
        prompt = RANKING_PROMPT.format(question=row["question"], options=options)
    elif row["type"] == "MultiSelectClassification":
        labels = _extract_allowed_labels(row)
        items = _extract_classification_items(row)
        items_block = "\n".join(f"- {item}" for item in items)
        labels_block = "\n".join(f"- {label}" for label in labels)
        prompt = MULTISELECT_CLASSIFICATION_PROMPT.format(
            question=row["question"],
            labels=labels_block,
            items=items_block,
        )
    else:
        options = ""
        option_keys = ["a", "b", "c", "d", "e", "f"]
        for key in option_keys:
            if row[key]:
                options+=f"{key}: {row[key]}"
        if row["type"] == "ImageMultipleChoice":
            prompt = [
                {
                    "type": "text",
                    "text": MULTIPLECHOICE_PROMPT.format(question=row["question"], options=options),
                },
            ]
            for image_data in row["image"]:
                prompt.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}"  # image/png not image/jpeg
                        }
                    }
                )
        else:
            prompt = MULTIPLECHOICE_PROMPT.format(question=row["question"], options=options)

    response = agent.run(prompt)
    if _is_valid_answer_format(row, response.answer):
        return response

    correction_prompt = f"""
Your previous JSON had an invalid answer format.
Return corrected JSON only with the same schema and a strictly valid answer string.

Original prompt:
{prompt}
"""
    corrected = agent.run(correction_prompt)
    if _is_valid_answer_format(row, corrected.answer):
        return corrected
    return response


class CyberQuizPipeline:
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
            processor = CyberQuizPreprocessor(self.file_path)
            df = processor.run()
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
            normalized_answer = _normalize_answer_for_scoring(row, resp.answer)
            normalized_ground_truth = _normalize_answer_for_scoring(row, row.get("ground_truth", ""))
            is_correct = _is_correct_answer(row, resp.answer)
            records.append(
                {
                    "qid": row["qid"],
                    "question_type": row["type"],
                    "question": row["question"],
                    "ground_truth": row["ground_truth"],
                    "normalized_ground_truth": normalized_ground_truth,
                    "model": model_name,
                    "answer": resp.answer,
                    "normalized_answer": normalized_answer,
                    "is_correct": is_correct,
                    "confidence": resp.confidence,
                    "reasoning": resp.reasoning,
                    "latency": resp.latency,
                    "a": row["a"], 
                    "b": row["b"],  
                    "c": row["c"], 
                    "d": row["d"], 
                }
            )
        return pd.DataFrame(records)

    def _save_results(self, results_df: pd.DataFrame) -> Path:
        """Write per-question CSV and aggregate summary JSON to a timestamped folder."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RESULTS_DIR / f"cyber_quiz_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # ---------- Per-question detail ----------
        detail_path = run_dir / "detailed_results.csv"
        results_df.to_csv(detail_path, index=False)
        print(f"  Detailed results → {detail_path}")

        # ---------- Per-model summary ----------
        # summary = self._compute_summary(results_df)
        # summary_path = run_dir / "summary.json"
        # with open(summary_path, "w") as f:
        #     json.dump(summary, f, indent=2)
        # print(f"  Summary           → {summary_path}")

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
        viz = CyberQuizVisualizer(results_df, run_dir)
        viz.generate_all()

