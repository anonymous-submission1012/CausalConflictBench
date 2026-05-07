from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from testCausal.sqa_stage3_conflict_ca_cot_metrics import group_judge_metrics, summarize_judge_records


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
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


def write_ca_cot_judge_report_bundle(
    output_dir: Path | str,
    records: List[Dict[str, Any]],
    judge_config: Dict[str, Any],
    run_config: Dict[str, Any],
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    summary = summarize_judge_records(records)
    bundle = {
        "summary": summary,
        "judge_config": judge_config,
        "run_config": run_config,
    }
    _write_json(output_path / "judge_summary.json", bundle)
    _write_json(output_path / "judge_config.json", judge_config)
    _write_jsonl(output_path / "judge_records.jsonl", records)
    _write_csv(output_path / "judge_metrics_by_conflict_intensity.csv", group_judge_metrics(records, "conflict_intensity"))
    _write_csv(output_path / "judge_metrics_by_causal_type.csv", group_judge_metrics(records, "causal_type"))
    _write_markdown_report(output_path / "ca_cot_report.md", summary, judge_config, run_config)
    return bundle


def _write_markdown_report(
    path: Path,
    summary: Dict[str, Any],
    judge_config: Dict[str, Any],
    run_config: Dict[str, Any],
) -> None:
    lines = [
        "# CA-CoT Judge Report",
        "",
        "## Run",
        f"- Base model: `{run_config.get('model', '')}`",
        f"- Prompt mode: `{run_config.get('prompt_mode', '')}`",
        f"- Judge model: `{judge_config.get('judge_model', '')}`",
        "",
        "## Coverage",
        f"- Total records: `{summary.get('total_records', 0)}`",
        f"- Judge OK records: `{summary.get('judge_ok_records', 0)}`",
        f"- Judge coverage: `{summary.get('judge_coverage', 0.0)}%`",
        "",
        "## Rule Recognition",
        f"- CD-Score avg: `{summary.get('cd_score_avg', 0.0)}%`",
        f"- Rule exact match count: `{summary.get('rule_exact_match_count', 0)}`",
        f"- Rule exact match rate: `{summary.get('rule_exact_match_rate', 0.0)}%`",
        f"- Rule conflict detected count: `{summary.get('rule_conflict_detected_count', 0)}`",
        f"- Rule conflict detected rate: `{summary.get('rule_conflict_detected_rate', 0.0)}%`",
        "",
        "## Quadrants",
        f"- Quadrant A: `{summary.get('quadrant_A', 0)}`",
        f"- Quadrant B: `{summary.get('quadrant_B', 0)}`",
        f"- Quadrant C: `{summary.get('quadrant_C', 0)}`",
        f"- Quadrant D: `{summary.get('quadrant_D', 0)}`",
        "",
        "## Error Types",
        f"- Type I: `{summary.get('type_i', 0)}`",
        f"- Type II: `{summary.get('type_ii', 0)}`",
        f"- Undetermined / other D failures: `{summary.get('undetermined', 0)}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
