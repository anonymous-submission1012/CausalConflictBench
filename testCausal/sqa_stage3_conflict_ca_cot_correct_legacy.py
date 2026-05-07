from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from testCausal.sqa_stage3_conflict_ca_cot_judge import derive_error_type
from testCausal.sqa_stage3_conflict_ca_cot_report import write_ca_cot_judge_report_bundle
from testCausal.sqa_stage3_conflict_dataset import load_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Correct legacy CA-CoT judge artifacts by backfilling matches_factual_baseline and recomputing error_type."
    )
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_predictions_map(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    predictions = load_jsonl(run_dir / "predictions.jsonl")
    return {str(row.get("sample_id", "")).strip(): row for row in predictions if str(row.get("sample_id", "")).strip()}


def _correct_records(records: Iterable[Dict[str, Any]], predictions_by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    corrected: List[Dict[str, Any]] = []
    for record in records:
        row = dict(record)
        sample_id = str(row.get("sample_id", "")).strip()
        prediction = predictions_by_id.get(sample_id, {})
        matches_factual_baseline = prediction.get("matches_factual_baseline")
        row["matches_factual_baseline"] = matches_factual_baseline if isinstance(matches_factual_baseline, bool) else None
        if "error_type" in row:
            row["error_type_legacy"] = row.get("error_type")
        if row.get("judge_status") == "ok":
            row["error_type"] = derive_error_type(
                bool(row.get("is_correct")),
                float(row.get("cd_score", 0.0)),
                row["matches_factual_baseline"],
            )
        corrected.append(row)
    return corrected


def run_correction(args: argparse.Namespace) -> Dict[str, Any]:
    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir does not exist: {source_dir}")

    run_dir = source_dir.parent
    output_dir = Path(args.output_dir) if args.output_dir else source_dir.with_name(source_dir.name + "_corrected")
    output_dir.mkdir(parents=True, exist_ok=True)

    judge_config = _load_json(source_dir / "judge_config.json")
    run_config = _load_json(run_dir / "run_config.json")
    source_records = list(load_jsonl(source_dir / "judge_records.jsonl"))
    predictions_by_id = _load_predictions_map(run_dir)
    corrected_records = _correct_records(source_records, predictions_by_id)

    bundle = write_ca_cot_judge_report_bundle(
        output_dir=output_dir,
        records=corrected_records,
        judge_config=judge_config,
        run_config=run_config,
    )
    return {
        "output_dir": str(output_dir),
        "summary": bundle["summary"],
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = run_correction(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
