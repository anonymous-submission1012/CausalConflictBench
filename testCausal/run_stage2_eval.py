from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from testCausal.sqa_stage2_dataset import load_jsonl, load_stage2_samples
from testCausal.sqa_stage2_report import write_report_bundle
from testCausal.sqa_stage2_runner import ensure_output_dir, run_stage2_evaluation, sanitize_args_for_logging


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def split_items_for_worker(items: Sequence[Any], worker_index: int, parallelism: int) -> list[Any]:
    if parallelism < 1:
        raise ValueError("parallelism must be >= 1")
    if worker_index < 0 or worker_index >= parallelism:
        raise ValueError("worker_index must satisfy 0 <= worker_index < parallelism")
    return [item for index, item in enumerate(items) if index % parallelism == worker_index]


def load_selected_samples(args: Any) -> tuple[list[Any], dict[str, Any]]:
    return load_stage2_samples(
        data_file=args.data_file,
        image_root=args.image_root,
        split=None if args.split == "all" else args.split,
        limit=args.limit,
        topics=args.topic,
        categories=args.category,
        causal_types=args.causal_type,
        qids=args.qids,
    )


def resolve_worker_qids(args: Any) -> list[str]:
    samples, _ = load_selected_samples(args)
    worker_samples = split_items_for_worker(samples, args.worker_index, args.parallelism)
    return [sample.qid for sample in worker_samples]


def build_worker_output_dir(output_dir: Path, parallelism: int, worker_index: int) -> Path:
    return output_dir / "workers" / f"p{parallelism}_worker_{worker_index:02d}"


def build_worker_command(base_argv: Sequence[str], worker_index: int, parallelism: int, worker_output_dir: Path) -> list[str]:
    filtered_args: list[str] = []
    skip_next = False
    stripped_flags = {"--parallelism", "--worker_index", "--output_dir"}
    for token in base_argv:
        if skip_next:
            skip_next = False
            continue
        if token in stripped_flags:
            skip_next = True
            continue
        if any(token.startswith(f"{flag}=") for flag in stripped_flags):
            continue
        filtered_args.append(token)

    return [
        sys.executable,
        str(CURRENT_DIR / "run_stage2_eval.py"),
        *filtered_args,
        "--parallelism",
        str(parallelism),
        "--worker_index",
        str(worker_index),
        "--output_dir",
        str(worker_output_dir),
    ]


def merge_worker_outputs(output_dir: Path, worker_count: int, samples: Sequence[Any]) -> list[dict[str, Any]]:
    records_by_qid: dict[str, dict[str, Any]] = {}
    for worker_index in range(worker_count):
        worker_dir = build_worker_output_dir(output_dir, worker_count, worker_index)
        predictions_path = worker_dir / "predictions.jsonl"
        for record in load_jsonl(predictions_path):
            records_by_qid[str(record["qid"])] = record

    missing_qids = [sample.qid for sample in samples if sample.qid not in records_by_qid]
    if missing_qids:
        raise ValueError(f"Missing merged predictions for qids: {missing_qids[:20]}")

    return [records_by_qid[sample.qid] for sample in samples]


def write_predictions_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 2 causal evaluation on ScienceQA-derived benchmark data.")
    parser.add_argument(
        "--data_file",
        type=str,
        default=str(PROJECT_ROOT / "data" / "scienceqa" / "problems_step2_final_v2.json"),
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=str(PROJECT_ROOT / "data" / "scienceqa" / "images"),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(PROJECT_ROOT / "results"),
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--label", type=str, default="stage2_eval")

    parser.add_argument("--split", type=str, default="test", choices=["all", "train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--qids", nargs="+", default=None)
    parser.add_argument("--topic", nargs="+", default=None)
    parser.add_argument("--category", nargs="+", default=None)
    parser.add_argument("--causal_type", nargs="+", default=None, choices=["effect", "cause"])

    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--api_base_url", type=str, default="<LOCAL_API_BASE_URL_PLACEHOLDER>")
    parser.add_argument("--endpoint_type", type=str, default="responses", choices=["responses", "chat_completions"])
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_key_env", type=str, default="INFINITEAI_API_KEY")

    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="answer_only_extractable",
        choices=["answer_only_extractable", "direct_mcq", "cot_mcq"],
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--image_detail", type=str, default="auto")
    parser.add_argument("--reasoning_effort", type=str, default=None)
    parser.add_argument("--user_agent", type=str, default="ScienceQA-Stage2-Eval/1.0")
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--store", action="store_true")
    parser.add_argument("--parallelism", type=positive_int, default=1)
    parser.add_argument("--worker_index", type=int, default=None, help=argparse.SUPPRESS)
    return parser


def run_parallel_stage2_evaluation(args: Any, base_argv: Sequence[str]) -> dict[str, Any]:
    output_dir = ensure_output_dir(args)
    samples, preflight_report = load_selected_samples(args)
    if not samples:
        raise ValueError("No valid samples matched the current filters.")

    processes: list[tuple[int, subprocess.Popen[Any]]] = []
    for worker_index in range(args.parallelism):
        worker_samples = split_items_for_worker(samples, worker_index, args.parallelism)
        if not worker_samples:
            continue
        worker_output_dir = build_worker_output_dir(output_dir, args.parallelism, worker_index)
        command = build_worker_command(base_argv, worker_index, args.parallelism, worker_output_dir)
        processes.append((worker_index, subprocess.Popen(command, cwd=str(PROJECT_ROOT))))

    failed_workers: list[tuple[int, int, Sequence[str] | str]] = []
    for worker_index, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failed_workers.append((worker_index, return_code, process.args))

    if failed_workers:
        failed_worker_index, return_code, command = failed_workers[0]
        raise subprocess.CalledProcessError(
            returncode=return_code,
            cmd=command,
            output=f"Parallel worker {failed_worker_index} failed.",
        )

    merged_records = merge_worker_outputs(output_dir, args.parallelism, samples)
    write_predictions_jsonl(output_dir / "predictions.jsonl", merged_records)

    run_config = sanitize_args_for_logging(args)
    run_config["output_dir"] = str(output_dir)
    report_bundle = write_report_bundle(output_dir, merged_records, preflight_report, run_config)
    return {
        "output_dir": str(output_dir),
        "summary": report_bundle["summary"],
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.worker_index is not None:
        args.qids = resolve_worker_qids(args)
    print("====Input Arguments====")
    print(json.dumps({**vars(args), "api_key": "***" if args.api_key else None}, indent=2, ensure_ascii=False))
    if args.parallelism > 1 and not args.dry_run and args.worker_index is None:
        result = run_parallel_stage2_evaluation(args, sys.argv[1:])
    else:
        result = run_stage2_evaluation(args)
    print("====Run Result====")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
