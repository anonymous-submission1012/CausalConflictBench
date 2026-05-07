from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.inference_api import APIClientError, InferenceAPIClient
from test_visual_causal.vci_eval_dataset import load_vci_samples
from test_visual_causal.vci_eval_runner import resolve_api_key


def build_description_prompt() -> str:
    return (
        "Describe only the visible content of this image in 3-6 concise bullet points. "
        "Focus on objects, layouts, colors, shapes, arrows, labels, and relative visual changes. "
        "Do not infer hidden causes, scientific rules, answers, or explanations beyond what is directly visible."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe image-description behavior across VCI endpoints.")
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(PROJECT_ROOT / "data" / "VCI"),
    )
    parser.add_argument("--question_subdir", type=str, default="shuffled_no_because")
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--sample_key", type=str, default=None)

    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--api_base_url", type=str, default="<LOCAL_API_BASE_URL_PLACEHOLDER>")
    parser.add_argument(
        "--endpoint_type",
        type=str,
        default="both",
        choices=["both", "chat_completions", "responses"],
    )
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--prompt_text", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--image_detail", type=str, default="auto")
    parser.add_argument("--reasoning_effort", type=str, default=None)
    parser.add_argument("--user_agent", type=str, default="VCI-Image-Probe/1.0")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--store", action="store_true")
    return parser


def resolve_endpoint_types(endpoint_type: str) -> list[str]:
    if endpoint_type == "both":
        return ["chat_completions", "responses"]
    return [endpoint_type]


def _apply_limit(items: list[dict[str, Any]], limit: Any) -> list[dict[str, Any]]:
    if limit is None:
        return items
    parsed = int(limit)
    if parsed <= 0:
        return []
    return items[:parsed]


def _resolve_probe_items_from_domain(args: Any) -> list[dict[str, Any]]:
    samples, _ = load_vci_samples(
        data_root=getattr(args, "data_root", str(PROJECT_ROOT / "data" / "VCI")),
        question_subdir=getattr(args, "question_subdir", "shuffled_no_because"),
        domains=[str(args.domain)],
    )

    seen_image_paths: set[Path] = set()
    items: list[dict[str, Any]] = []
    for sample in samples:
        image_path = Path(sample.image_path).resolve()
        if image_path in seen_image_paths:
            continue
        seen_image_paths.add(image_path)
        items.append({"sample_key": sample.sample_key, "image_path": image_path})
    return _apply_limit(items, getattr(args, "limit", None))


def _resolve_probe_items_from_image_dir(args: Any) -> list[dict[str, Any]]:
    image_dir = Path(str(args.image_dir)).expanduser().resolve()
    if not image_dir.is_dir():
        raise ValueError(f"image_dir does not exist or is not a directory: {image_dir}")

    supported_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    items = [
        {"sample_key": None, "image_path": path.resolve()}
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in supported_suffixes
    ]
    return _apply_limit(items, getattr(args, "limit", None))


def resolve_probe_items(args: Any) -> list[dict[str, Any]]:
    if getattr(args, "image_path", None) or getattr(args, "sample_key", None):
        image_path = resolve_image_path(args)
        sample_key = getattr(args, "sample_key", None)
        return [{"sample_key": sample_key, "image_path": image_path}]

    if getattr(args, "image_dir", None):
        return _resolve_probe_items_from_image_dir(args)

    if getattr(args, "domain", None):
        return _resolve_probe_items_from_domain(args)

    raise ValueError("Provide one of --image_path, --sample_key, --domain, or --image_dir.")


def resolve_image_path(args: Any) -> Path:
    if getattr(args, "image_path", None):
        return Path(str(args.image_path)).expanduser().resolve()

    raw_sample_key = getattr(args, "sample_key", None)
    sample_key = str(raw_sample_key).strip() if raw_sample_key is not None else ""
    if not sample_key:
        raise ValueError("Provide either --image_path or --sample_key.")

    samples, _ = load_vci_samples(
        data_root=getattr(args, "data_root", str(PROJECT_ROOT / "data" / "VCI")),
        question_subdir=getattr(args, "question_subdir", "shuffled_no_because"),
        sample_keys=[sample_key],
    )
    if not samples:
        raise ValueError(f"No VCI sample matched sample_key={sample_key!r}.")
    return Path(samples[0].image_path).resolve()


def run_probe_once(image_path: Path, endpoint_type: str, args: Any) -> str:
    api_key, _ = resolve_api_key(getattr(args, "api_key", None), getattr(args, "api_key_env", None))
    client = InferenceAPIClient(
        base_url=args.api_base_url,
        endpoint_type=endpoint_type,
        model=args.model,
        api_key=api_key,
        timeout=args.timeout,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_tokens,
        stream=args.stream,
        store=getattr(args, "store", False),
        reasoning_effort=getattr(args, "reasoning_effort", None),
        image_detail=args.image_detail,
        user_agent=args.user_agent,
    )
    raw_prompt_text = getattr(args, "prompt_text", None)
    prompt_text = str(raw_prompt_text).strip() if raw_prompt_text is not None else ""
    prompt_text = prompt_text or build_description_prompt()
    return client.generate(
        system_message=None,
        content_blocks=[
            {"type": "text", "text": prompt_text},
            {"type": "image_path", "image_path": str(image_path)},
        ],
    )


def run_probe_report(image_path: Path, args: Any) -> dict[str, dict[str, str]]:
    report: dict[str, dict[str, str]] = {}
    for endpoint_type in resolve_endpoint_types(args.endpoint_type):
        try:
            report[endpoint_type] = {
                "raw_output": run_probe_once(image_path=image_path, endpoint_type=endpoint_type, args=args),
            }
        except APIClientError as exc:
            report[endpoint_type] = {"error": str(exc)}
    return report


def run_batch_probe_report(items: list[dict[str, Any]], args: Any) -> list[dict[str, Any]]:
    report_items: list[dict[str, Any]] = []
    for item in items:
        image_path = Path(item["image_path"])
        report_items.append(
            {
                "sample_key": item.get("sample_key"),
                "image_path": str(image_path),
                "outputs": run_probe_report(image_path, args),
            }
        )
    return report_items


def write_report_json(output_path: Path | str, payload: dict[str, Any]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    probe_items = resolve_probe_items(args)
    prompt_text = (str(args.prompt_text).strip() if args.prompt_text is not None else "") or build_description_prompt()
    report = {
        "model": args.model,
        "endpoint_type": args.endpoint_type,
        "prompt_text": prompt_text,
        "total_images": len(probe_items),
        "items": run_batch_probe_report(probe_items, args),
    }
    if args.output_json:
        write_report_json(args.output_json, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
