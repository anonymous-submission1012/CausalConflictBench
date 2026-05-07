from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from test_visual_causal.vci_eval_metrics import group_metrics, summarize_records


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

    for group_field in ("domain", "question_scope", "transfer_type", "mechanism_id", "variant_id", "answer_prior"):
        _write_csv(output_path / f"metrics_by_{group_field}.csv", group_metrics(records, group_field))

    error_rows = [row for row in records if row.get("status") != "ok" or not row.get("is_correct")]
    _write_jsonl(output_path / "error_cases.jsonl", error_rows)
    _write_markdown_report(output_path / "report.md", summary, preflight_report, run_config)
    return bundle


def _write_markdown_report(
    path: Path,
    summary: Dict[str, Any],
    preflight_report: Dict[str, Any],
    run_config: Dict[str, Any],
) -> None:
    lines = [
        "# VCI Evaluation Report",
        "",
        "## Summary",
        f"- Model: `{run_config.get('model', '')}`",
        f"- Prompt mode: `{run_config.get('prompt_mode', '')}`",
        f"- Input modality: `{run_config.get('input_modality', 'multimodal')}`",
        f"- Text context source: `{run_config.get('text_context_source', 'visual_context')}`",
        f"- Total records: `{summary.get('total_records', 0)}`",
        f"- OK records: `{summary.get('ok_records', 0)}`",
        f"- Correct records: `{summary.get('correct_records', 0)}`",
        f"- Accuracy: `{summary.get('accuracy', 0.0)}%`",
        f"- Coverage: `{summary.get('coverage', 0.0)}%`",
        f"- Prior bias rate: `{summary.get('prior_bias_rate', 0.0)}%`",
        f"- Variant strict acc: `{summary.get('variant_strict_acc', 0.0)}%`",
        "",
        "## Preflight",
        f"- Valid samples: `{preflight_report.get('valid_sample_count', 0)}`",
        f"- Invalid samples skipped: `{preflight_report.get('invalid_sample_count', 0)}`",
        f"- Excluded samples skipped: `{preflight_report.get('excluded_sample_count', 0)}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
