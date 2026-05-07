from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test_visual_causal.vci_cot_analysis_runner import run_vci_cot_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VCI CoT post-analysis on an existing predictions.jsonl file.")
    parser.add_argument("--predictions_path", type=str, default=None)
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=str(PROJECT_ROOT / "data" / "VCI"))
    parser.add_argument("--question_subdir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--judge_label", type=str, default=None)
    parser.add_argument("--qids", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--analysis_mode", type=str, default="flip_rule", choices=["flip_rule", "rule_source", "both"])

    parser.add_argument("--judge_model", type=str, default="gpt-5.4")
    parser.add_argument("--api_base_url", type=str, default="<API_BASE_URL_PLACEHOLDER>")
    parser.add_argument("--endpoint_type", type=str, default="responses", choices=["responses", "chat_completions"])
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--reasoning_effort", type=str, default=None)
    parser.add_argument("--user_agent", type=str, default="VCI-CoT-Analysis/1.0")
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--store", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.predictions_path and not args.run_dir:
        parser.error("either --predictions_path or --run_dir is required")
    print("====Input Arguments====")
    print(json.dumps({**vars(args), "api_key": "***" if args.api_key else None}, indent=2, ensure_ascii=False))
    result = run_vci_cot_analysis(args)
    print("====Run Result====")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
