from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from models.inference_api import APIClientError, InferenceAPIClient
from testCausal.sqa_stage2_dataset import (
    Stage2Sample,
    append_jsonl,
    load_jsonl,
    load_stage2_samples,
    save_json,
)
from testCausal.sqa_stage2_prompt import build_prompt
from testCausal.sqa_stage2_report import write_report_bundle


def resolve_api_key(api_key: Optional[str], api_key_env: Optional[str], allow_dummy: bool = False) -> Tuple[str, str]:
    if api_key:
        return api_key, "cli"

    candidate_names = [api_key_env, "INFINITEAI_API_KEY", "OPENAI_API_KEY"]
    for name in candidate_names:
        if not name:
            continue
        value = os.getenv(name)
        if value:
            return value, f"env:{name}"

    if allow_dummy:
        return "DUMMY_KEY", "dummy"
    raise ValueError("API key not found. Use --api_key or set the configured environment variable.")


def create_client(args: Any) -> InferenceAPIClient:
    api_key, api_key_source = resolve_api_key(args.api_key, args.api_key_env, allow_dummy=args.dry_run)
    args.api_key_source = api_key_source
    return InferenceAPIClient(
        base_url=args.api_base_url,
        endpoint_type=args.endpoint_type,
        model=args.model,
        api_key=api_key,
        timeout=args.timeout,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_tokens,
        stream=args.stream,
        store=args.store,
        reasoning_effort=args.reasoning_effort,
        image_detail=args.image_detail,
        user_agent=args.user_agent,
    )


def redact_payload_for_preview(payload: Any) -> Any:
    if isinstance(payload, dict):
        redacted: Dict[str, Any] = {}
        for key, value in payload.items():
            if key == "image_url" and isinstance(value, str) and value.startswith("data:"):
                redacted[key] = "<data-url omitted>"
            elif key == "image_url" and isinstance(value, dict):
                nested = dict(value)
                if isinstance(nested.get("url"), str) and nested["url"].startswith("data:"):
                    nested["url"] = "<data-url omitted>"
                redacted[key] = redact_payload_for_preview(nested)
            else:
                redacted[key] = redact_payload_for_preview(value)
        return redacted
    if isinstance(payload, list):
        return [redact_payload_for_preview(item) for item in payload]
    return payload


def normalize_choice_text(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\"'`“”‘’]", "", text)
    text = re.sub(r"[.,;:!?()\[\]{}]", "", text)
    return text.strip()


def extract_option_letter(text: str, valid_options: Sequence[str]) -> Optional[str]:
    upper_text = text.upper()
    patterns = [
        r"THE CORRECT ANSWER IS\s*\(?([A-Z])\)?",
        r"THE ANSWER IS\s*\(?([A-Z])\)?",
        r"CORRECT ANSWER\s*:\s*\(?([A-Z])\)?",
        r"FINAL ANSWER\s*:\s*\(?([A-Z])\)?",
        r"ANSWER\s*:\s*\(?([A-Z])\)?",
        r"OPTION\s*\(?([A-Z])\)?",
        r"CHOICE\s*\(?([A-Z])\)?",
        r"^\s*\(?([A-Z])\)?(?:[\s\.:]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper_text, flags=re.MULTILINE)
        if match:
            candidate = match.group(1)
            if candidate in valid_options:
                return candidate
    return None


def extract_choice_text_match(text: str, choices: Sequence[str]) -> Optional[int]:
    normalized_output = normalize_choice_text(text)
    if not normalized_output:
        return None

    if "answer:" in normalized_output:
        normalized_output = normalized_output.split("answer:", 1)[1].strip()

    for index, choice in enumerate(choices):
        normalized_choice = normalize_choice_text(choice)
        if normalized_choice and normalized_choice in normalized_output:
            return index

    for index, choice in enumerate(choices):
        normalized_choice = normalize_choice_text(choice)
        if normalized_choice and normalized_output == normalized_choice:
            return index
    return None


def parse_prediction(text: str, sample: Stage2Sample) -> Tuple[Optional[int], str]:
    option_letter = extract_option_letter(text, sample.option_labels)
    if option_letter is not None:
        return sample.option_labels.index(option_letter), "option_letter"

    choice_idx = extract_choice_text_match(text, sample.choices)
    if choice_idx is not None:
        return choice_idx, "choice_text"

    return None, "unparsed"


def build_content_blocks(prompt_text: str, sample: Stage2Sample) -> List[Dict[str, str]]:
    return [
        {"type": "text", "text": prompt_text},
        {"type": "image_path", "image_path": str(sample.image_path)},
    ]


def run_single_inference(client: InferenceAPIClient, prompt_text: str, system_message: str, sample: Stage2Sample, args: Any) -> str:
    last_error: Optional[APIClientError] = None
    content_blocks = build_content_blocks(prompt_text, sample)
    for attempt in range(args.max_retries + 1):
        try:
            return client.generate(
                prompt_text=prompt_text,
                system_message=system_message,
                image_path=str(sample.image_path),
                content_blocks=content_blocks,
            )
        except APIClientError as exc:
            last_error = exc
            if attempt >= args.max_retries:
                break
    if last_error is None:
        raise RuntimeError("Inference failed without a captured exception.")
    raise last_error


def record_from_success(
    sample: Stage2Sample,
    output_text: str,
    pred_idx: int,
    prediction_source: str,
) -> Dict[str, Any]:
    pred_label = sample.option_labels[pred_idx]
    gold_label = sample.option_labels[sample.answer_idx]
    return {
        "qid": sample.qid,
        "split": sample.split,
        "topic": sample.topic,
        "category": sample.category,
        "causal_type": sample.causal_type,
        "choice_count": len(sample.choices),
        "has_choice_images": sample.has_choice_images,
        "stage2_quality_action": sample.stage2_quality_action,
        "image_path": str(sample.image_path),
        "status": "ok",
        "gold_index": sample.answer_idx,
        "gold_label": gold_label,
        "gold_choice": sample.answer_text,
        "pred_index": pred_idx,
        "pred_label": pred_label,
        "pred_choice": sample.choices[pred_idx],
        "prediction_source": prediction_source,
        "is_correct": pred_idx == sample.answer_idx,
        "raw_output": output_text,
    }


def record_from_error(sample: Stage2Sample, status: str, error_message: str, raw_output: str = "") -> Dict[str, Any]:
    return {
        "qid": sample.qid,
        "split": sample.split,
        "topic": sample.topic,
        "category": sample.category,
        "causal_type": sample.causal_type,
        "choice_count": len(sample.choices),
        "has_choice_images": sample.has_choice_images,
        "stage2_quality_action": sample.stage2_quality_action,
        "image_path": str(sample.image_path),
        "status": status,
        "error_message": error_message,
        "is_correct": False,
        "raw_output": raw_output,
    }


def load_existing_records(predictions_path: Path) -> Dict[str, Dict[str, Any]]:
    latest_records: Dict[str, Dict[str, Any]] = {}
    for record in load_jsonl(predictions_path):
        latest_records[str(record["qid"])] = record
    return latest_records


def sanitize_args_for_logging(args: Any) -> Dict[str, Any]:
    payload = vars(args).copy()
    if payload.get("api_key"):
        payload["api_key"] = "***"
    return payload


def ensure_output_dir(args: Any) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model)
        output_dir = Path(args.output_root) / "stage2_eval" / safe_model / args.label
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def print_dry_run_preview(client: InferenceAPIClient, sample: Stage2Sample, prompt_text: str, system_message: str) -> None:
    payload = client.build_payload(
        prompt_text=prompt_text,
        system_message=system_message,
        image_path=str(sample.image_path),
        content_blocks=build_content_blocks(prompt_text, sample),
    )
    print("====Dry Run Prompt====")
    print(prompt_text)
    print("")
    print("====Dry Run System Message====")
    print(system_message)
    print("")
    print("====Dry Run Payload (redacted)====")
    print(json.dumps(redact_payload_for_preview(payload), indent=2, ensure_ascii=False))


def run_stage2_evaluation(args: Any) -> Dict[str, Any]:
    random.seed(args.seed)
    output_dir = ensure_output_dir(args)

    samples, preflight_report = load_stage2_samples(
        data_file=args.data_file,
        image_root=args.image_root,
        split=None if args.split == "all" else args.split,
        limit=args.limit,
        topics=args.topic,
        categories=args.category,
        causal_types=args.causal_type,
        qids=args.qids,
    )
    save_json(output_dir / "preflight_report.json", preflight_report)

    if not samples:
        raise ValueError("No valid samples matched the current filters.")

    client = create_client(args)
    prompt_text, system_message = build_prompt(samples[0], prompt_mode=args.prompt_mode)
    if args.dry_run:
        print_dry_run_preview(client, samples[0], prompt_text, system_message)
        return {
            "dry_run": True,
            "sample_qid": samples[0].qid,
            "output_dir": str(output_dir),
        }

    predictions_path = output_dir / "predictions.jsonl"
    existing_records = load_existing_records(predictions_path) if args.resume else {}
    records_by_qid: Dict[str, Dict[str, Any]] = dict(existing_records)
    completed_qids = {
        qid for qid, record in existing_records.items() if str(record.get("status")) == "ok"
    }

    for index, sample in enumerate(samples, start=1):
        if sample.qid in completed_qids:
            continue

        prompt_text, system_message = build_prompt(sample, prompt_mode=args.prompt_mode)
        try:
            output_text = run_single_inference(client, prompt_text, system_message, sample, args)
        except APIClientError as exc:
            record = record_from_error(sample, status="api_error", error_message=str(exc))
        else:
            pred_idx, prediction_source = parse_prediction(output_text, sample)
            if pred_idx is None:
                record = record_from_error(
                    sample,
                    status="parse_error",
                    error_message="Unable to map model output to an option.",
                    raw_output=output_text,
                )
            else:
                record = record_from_success(sample, output_text, pred_idx, prediction_source)

        append_jsonl(predictions_path, [record])
        records_by_qid[sample.qid] = record
        if record["status"] == "ok":
            completed_qids.add(sample.qid)

        if args.debug or index <= 3:
            print(f"[{index}/{len(samples)}] qid={sample.qid} status={record['status']} correct={record.get('is_correct')}")

    run_config = sanitize_args_for_logging(args)
    run_config["output_dir"] = str(output_dir)
    ordered_records = [
        records_by_qid[sample.qid]
        for sample in samples
        if sample.qid in records_by_qid
    ]
    report_bundle = write_report_bundle(output_dir, ordered_records, preflight_report, run_config)
    return {
        "output_dir": str(output_dir),
        "summary": report_bundle["summary"],
    }
