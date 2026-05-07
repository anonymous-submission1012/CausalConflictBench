from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from testCausal.sqa_stage3_conflict_dataset import (  # noqa: E402
    Stage3ConflictSample,
    load_jsonl,
    load_stage3_conflict_samples,
)
from testCausal.sqa_stage3_conflict_report import write_report_bundle  # noqa: E402


UTF8_BOM = b"\xef\xbb\xbf"


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as file_obj:
        payload = json.load(file_obj)
    return payload if isinstance(payload, dict) else {}


def _write_jsonl_preserving_bom(path: Path, records: Iterable[dict[str, Any]]) -> None:
    has_bom = path.exists() and path.read_bytes().startswith(UTF8_BOM)
    encoding = "utf-8-sig" if has_bom else "utf-8"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding=encoding, newline="\n") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def build_reasoning_complexity_index(samples: Sequence[Stage3ConflictSample]) -> dict[str, str]:
    return {
        sample.sample_id: sample.reasoning_complexity
        for sample in samples
        if sample.reasoning_complexity
    }


def _load_report_context(output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_bundle = _read_json_object(output_dir / "summary.json")
    preflight_report = summary_bundle.get("preflight")
    run_config = summary_bundle.get("run_config")

    if not isinstance(preflight_report, dict):
        preflight_report = _read_json_object(output_dir / "preflight_report.json")
    if not isinstance(run_config, dict):
        run_config = _read_json_object(output_dir / "run_config.json")

    return preflight_report, run_config


def backfill_output_dir_from_samples(
    output_dir: Path | str,
    samples: Sequence[Stage3ConflictSample],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    predictions_path = output_path / "predictions.jsonl"
    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions.jsonl not found: {predictions_path}")

    records = load_jsonl(predictions_path)
    reasoning_by_sample_id = build_reasoning_complexity_index(samples)
    missing_sample_ids: list[str] = []
    updated_records = 0

    for record in records:
        sample_id = str(record.get("sample_id", ""))
        reasoning_complexity = reasoning_by_sample_id.get(sample_id)
        if not reasoning_complexity:
            missing_sample_ids.append(sample_id)
            continue
        if record.get("reasoning_complexity") != reasoning_complexity:
            record["reasoning_complexity"] = reasoning_complexity
            updated_records += 1

    if missing_sample_ids:
        return {
            "output_dir": str(output_path),
            "total_records": len(records),
            "updated_records": updated_records,
            "missing_sample_ids": missing_sample_ids,
            "dry_run": dry_run,
            "status": "missing_sample_ids",
        }

    if not dry_run:
        _write_jsonl_preserving_bom(predictions_path, records)
        preflight_report, run_config = _load_report_context(output_path)
        write_report_bundle(output_path, records, preflight_report, run_config)

    return {
        "output_dir": str(output_path),
        "total_records": len(records),
        "updated_records": updated_records,
        "missing_sample_ids": [],
        "dry_run": dry_run,
        "status": "ok",
    }


def discover_output_dirs(root: Path | str) -> list[Path]:
    root_path = Path(root)
    if (root_path / "predictions.jsonl").exists():
        return [root_path]
    return sorted({path.parent for path in root_path.rglob("predictions.jsonl")})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill Stage 3 reasoning_complexity fields and regenerate reports.")
    parser.add_argument("output", nargs="+", help="Output directory or root containing predictions.jsonl files.")
    parser.add_argument("--data_file", default=str(PROJECT_ROOT / "data" / "scienceqa" / "problems_tro.json"))
    parser.add_argument("--image_root", default=str(PROJECT_ROOT / "data" / "scienceqa" / "images"))
    parser.add_argument("--captions_file", default=str(PROJECT_ROOT / "data" / "captions.json"))
    parser.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    samples, _ = load_stage3_conflict_samples(
        data_file=args.data_file,
        image_root=args.image_root,
        captions_file=args.captions_file,
        split=None if args.split == "all" else args.split,
        task_variant="conflict",
    )

    results = []
    for output in args.output:
        for output_dir in discover_output_dirs(output):
            results.append(backfill_output_dir_from_samples(output_dir, samples, dry_run=args.dry_run))

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
