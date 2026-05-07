from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from testCausal.sqa_stage3_conflict_metrics import group_metrics, group_metrics_by_fields, summarize_records


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report_bundle(
    output_dir: Path | str,
    records: List[Dict[str, Any]],
    preflight_report: Dict[str, Any],
    run_config: Dict[str, Any],
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    summary = summarize_records(records)
    bundle = {
        "summary": summary,
        "preflight": preflight_report,
        "run_config": run_config,
    }
    _write_json(output_path / "summary.json", bundle)
    _write_json(output_path / "run_config.json", run_config)
    _write_json(output_path / "preflight_report.json", preflight_report)

    group_fields = ["split", "topic", "category", "causal_type", "conflict_intensity"]
    if any(row.get("donor_topic") for row in records):
        group_fields.append("donor_topic")
    if any(row.get("donor_category") for row in records):
        group_fields.append("donor_category")

    for group_field in group_fields:
        _write_csv(output_path / f"metrics_by_{group_field}.csv", group_metrics(records, group_field))

    if any(row.get("reasoning_complexity") for row in records):
        _write_csv(output_path / "metrics_by_reasoning_complexity.csv", group_metrics(records, "reasoning_complexity"))
        _write_csv(
            output_path / "metrics_by_conflict_intensity_reasoning_complexity.csv",
            group_metrics_by_fields(records, ["conflict_intensity", "reasoning_complexity"]),
        )
        _write_csv(
            output_path / "metrics_by_causal_type_reasoning_complexity.csv",
            group_metrics_by_fields(records, ["causal_type", "reasoning_complexity"]),
        )

    error_rows = [row for row in records if row.get("status") != "ok" or not row.get("is_correct")]
    baseline_bias_rows = [row for row in records if row.get("matches_factual_baseline") is True]
    _write_jsonl(output_path / "error_cases.jsonl", error_rows)
    _write_jsonl(output_path / "baseline_bias_cases.jsonl", baseline_bias_rows)
    _write_markdown_report(output_path / "report.md", records, summary, preflight_report, run_config)
    return bundle


def _write_markdown_report(
    path: Path,
    records: List[Dict[str, Any]],
    summary: Dict[str, Any],
    preflight_report: Dict[str, Any],
    run_config: Dict[str, Any],
) -> None:
    task_variant = str(run_config.get("task_variant", "conflict"))
    lines = [
        "# Stage 3 Evaluation Report",
        "",
        "## Summary",
        f"- Task variant: `{task_variant}`",
        f"- Rule position: `{run_config.get('rule_position', 'prefix')}`",
        f"- Input modality: `{run_config.get('input_modality', 'multimodal')}`",
        f"- Text context source: `{run_config.get('text_context_source', 'caption')}`",
        f"- Model: `{run_config.get('model', '')}`",
        f"- Split: `{run_config.get('split', '')}`",
        f"- Prompt mode: `{run_config.get('prompt_mode', '')}`",
        f"- Total records: `{summary.get('total_records', 0)}`",
        f"- OK records: `{summary.get('ok_records', 0)}`",
        f"- Correct records: `{summary.get('correct_records', 0)}`",
        f"- Accuracy: `{summary.get('accuracy', 0.0)}%`",
        f"- QID-Acc: `{summary.get('qid_acc', 0.0)}%`",
        f"- Coverage: `{summary.get('coverage', 0.0)}%`",
        "",
        "## Preflight",
        f"- Valid samples: `{preflight_report.get('valid_sample_count', 0)}`",
        f"- Invalid samples skipped: `{preflight_report.get('invalid_sample_count', 0)}`",
        "",
        "## Status Counts",
    ]
    if "factual_baseline_bias_rate" in summary:
        lines.insert(10, f"- Factual baseline bias rate: `{summary.get('factual_baseline_bias_rate', 0.0)}%`")
    if any(row.get("reasoning_complexity") for row in records):
        lines.extend(
            [
                "",
                "## Reasoning Complexity",
                "- Available grouping file: `metrics_by_reasoning_complexity.csv`",
                "- Cross-analysis file: `metrics_by_conflict_intensity_reasoning_complexity.csv`",
                "- Cross-analysis file: `metrics_by_causal_type_reasoning_complexity.csv`",
            ]
        )
    for key, value in sorted(summary.get("status_counts", {}).items()):
        lines.append(f"- `{key}`: `{value}`")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
