from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from test_visual_causal.vci_cot_analysis_metrics import group_metrics, summarize_records


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as file_obj:
            file_obj.write("")
        return

    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
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
    judge_config: Dict[str, Any],
    run_config: Dict[str, Any],
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    summary = summarize_records(records)
    bundle = {
        "summary": summary,
        "judge_config": judge_config,
        "run_config": run_config,
    }
    _write_json(output_path / "cot_analysis_summary.json", bundle)
    _write_json(output_path / "cot_analysis_run_config.json", run_config)
    _write_json(output_path / "cot_analysis_judge_config.json", judge_config)

    for group_field in ("domain", "question_scope", "transfer_type", "mechanism_id", "variant_id"):
        _write_csv(output_path / f"cot_analysis_metrics_by_{group_field}.csv", group_metrics(records, group_field))

    focus_rows = [row for row in records if row.get("analysis_status") != "ok" or row.get("joint_bucket") != "cot_correct_answer_correct"]
    _write_jsonl(output_path / "cot_analysis.jsonl", records)
    _write_jsonl(output_path / "cot_analysis_focus_cases.jsonl", focus_rows)
    _write_markdown_report(output_path / "cot_analysis_report.md", summary, judge_config, run_config)
    return bundle


def _write_markdown_report(
    path: Path,
    summary: Dict[str, Any],
    judge_config: Dict[str, Any],
    run_config: Dict[str, Any],
) -> None:
    analysis_mode = str(run_config.get("analysis_mode", "flip_rule") or "flip_rule")
    lines = [
        "# VCI CoT Analysis Report",
        "",
        "## Summary",
        f"- Judge model: `{judge_config.get('judge_model', '')}`",
        f"- Source predictions: `{run_config.get('source_predictions_path', '')}`",
        f"- Question subdir: `{run_config.get('question_subdir', '')}`",
        f"- Total records: `{summary.get('total_records', 0)}`",
        f"- Analysis OK records: `{summary.get('analysis_ok_records', 0)}`",
        f"- Analysis coverage: `{summary.get('analysis_coverage', 0.0)}%`",
    ]

    if analysis_mode in {"flip_rule", "both"}:
        lines.extend(
            [
                "",
                "## Rule Match",
                f"- Rule judgeable records: `{summary.get('cot_rule_judgeable_records', 0)}`",
                f"- Rule accuracy: `{summary.get('cot_rule_accuracy', 0.0)}%`",
                f"- Cot correct + answer correct: `{summary.get('cot_correct_answer_correct', 0)}`",
                f"- Cot correct + answer wrong: `{summary.get('cot_correct_answer_wrong', 0)}`",
                f"- Cot wrong + answer correct: `{summary.get('cot_wrong_answer_correct', 0)}`",
                f"- Cot wrong + answer wrong: `{summary.get('cot_wrong_answer_wrong', 0)}`",
                f"- Cot undecidable: `{summary.get('cot_undecidable', 0)}`",
            ]
        )

    if analysis_mode in {"rule_source", "both"}:
        lines.extend(
            [
                "",
                "## Rule Source",
                f"- Rule-source factual_rule: `{summary.get('cot_rule_source_factual_rule', 0)}`",
                f"- Rule-source flipped_rule: `{summary.get('cot_rule_source_flipped_rule', 0)}`",
                f"- Rule-source other: `{summary.get('cot_rule_source_other', 0)}`",
                f"- Factual-rule ratio among source-labeled records: `{summary.get('cot_rule_source_factual_rule_ratio', 0.0)}%`",
                f"- Answer correct rate given factual_rule: `{summary.get('answer_correct_rate_given_factual_rule', 0.0)}%`",
                f"- Answer correct rate given flipped_rule: `{summary.get('answer_correct_rate_given_flipped_rule', 0.0)}%`",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
