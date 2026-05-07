from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from models.inference_api import APIClientError, InferenceAPIClient
from test_visual_causal.vci_eval_dataset import (
    VciSample,
    append_jsonl,
    build_sample_key,
    load_jsonl,
    load_sample_exclusion_spec,
    load_vci_samples,
    save_json,
)
from test_visual_causal.vci_eval_prompt import (
    build_prompt,
    get_vci_cot_required_steps,
    is_vci_cot_prompt_mode,
)
from test_visual_causal.vci_eval_report import write_report_bundle


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
    text = re.sub("[\\\"'`\\u201c\\u201d\\u2018\\u2019]", "", text)
    text = re.sub(r"[.,;:!?()\[\]{}]", "", text)
    return text.strip()


def extract_option_letter(text: str, valid_options: Sequence[str]) -> Optional[str]:
    upper_text = text.upper()
    stripped_text = upper_text.strip()
    if stripped_text in valid_options:
        return stripped_text

    lines = [line.strip() for line in upper_text.splitlines() if line.strip()]
    if lines and lines[-1] in valid_options:
        return lines[-1]

    if lines:
        explicit_last_line_patterns = [
            r"(?:FINAL ANSWER|ANSWER|CORRECT ANSWER)\s*(?::|IS)?\s*(?:OPTION|CHOICE)?\s*\(?([A-Z])\)?",
            r"(?:OPTION|CHOICE)\s*\(?([A-Z])\)?",
        ]
        for pattern in explicit_last_line_patterns:
            match = re.fullmatch(pattern, lines[-1])
            if match:
                candidate = match.group(1)
                if candidate in valid_options:
                    return candidate

    patterns = [
        r"THE CORRECT ANSWER IS\s*(?:OPTION|CHOICE)?\s*\(?([A-Z])\)?(?=\s|$|[.)!,:;])",
        r"THE ANSWER IS\s*(?:OPTION|CHOICE)?\s*\(?([A-Z])\)?(?=\s|$|[.)!,:;])",
        r"CORRECT ANSWER\s*:\s*(?:OPTION|CHOICE)?\s*\(?([A-Z])\)?(?=\s|$|[.)!,:;])",
        r"FINAL ANSWER\s*(?::|IS)\s*(?:OPTION|CHOICE)?\s*\(?([A-Z])\)?(?=\s|$|[.)!,:;])",
        r"ANSWER\s*:\s*(?:OPTION|CHOICE)?\s*\(?([A-Z])\)?(?=\s|$|[.)!,:;])",
        r"\\BOXED\{\s*([A-Z])\s*\}",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper_text)
        if match:
            candidate = match.group(1)
            if candidate in valid_options:
                return candidate
    return None


def extract_choice_text_match(text: str, choices: Sequence[str]) -> Optional[int]:
    normalized_text = text.replace("\r\n", "\n").strip()
    if not normalized_text:
        return None

    candidate_segments = [normalized_text]
    lines = [line.strip() for line in normalized_text.split("\n") if line.strip()]
    if lines:
        candidate_segments.append(lines[-1])
        if len(lines) >= 2:
            candidate_segments.append(lines[-2])

    normalized_segments: List[str] = []
    for segment in candidate_segments:
        normalized_segment = normalize_choice_text(segment)
        if not normalized_segment:
            continue
        normalized_segments.append(normalized_segment)

        anchored_match = re.search(
            r"(?:^| )(?:answer|final answer|correct answer|the answer is|the correct answer is|option|choice|答案)\s*[:：]?\s*(.+)$",
            normalized_segment,
        )
        if anchored_match:
            normalized_segments.append(anchored_match.group(1).strip())

    for index, choice in enumerate(choices):
        normalized_choice = normalize_choice_text(choice)
        if not normalized_choice:
            continue
        for segment in normalized_segments:
            if segment == normalized_choice:
                return index
    return None


def parse_prediction(text: str, sample: VciSample) -> Tuple[Optional[int], str]:
    option_letter = extract_option_letter(text, sample.option_labels)
    if option_letter is not None:
        return sample.option_labels.index(option_letter), "option_letter"

    choice_idx = extract_choice_text_match(text, sample.choices)
    if choice_idx is not None:
        return choice_idx, "choice_text"

    return None, "unparsed"


def extract_final_answer_region(text: str) -> str:
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return ""

    think_matches = list(re.finditer(r"(?i)</think>", normalized))
    if think_matches:
        trailing = normalized[think_matches[-1].end():].strip()
        if trailing:
            return trailing

    lines = [line.strip() for line in normalized.split("\n") if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-3:])


def extract_vci_cot_steps(text: str, prompt_mode: str) -> Dict[str, Any]:
    normalized_text = text.replace("\r\n", "\n").strip()
    matches = list(
        re.finditer(
            r"(?im)^[ \t>*-]*[A-Za-z]*step\s*([1-4])\s*:\s*(.*?)\s*(?:\*+)?\s*$",
            normalized_text,
        )
    )
    step_payload: Dict[str, Any] = {}
    if not matches:
        step_payload["vci_cot_has_complete_steps"] = False
        return step_payload

    extracted_steps: Dict[str, str] = {}
    for index, match in enumerate(matches):
        step_number = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized_text)
        content = normalized_text[start:end].strip()
        extracted_steps[f"vci_cot_step_{step_number}"] = content

    step_payload.update(extracted_steps)
    required_steps = get_vci_cot_required_steps(prompt_mode)
    step_payload["vci_cot_has_complete_steps"] = all(
        bool(step_payload.get(f"vci_cot_step_{step_number}")) for step_number in required_steps
    )
    return step_payload


def backfill_vci_cot_record(record: Dict[str, Any], prompt_mode: str) -> Dict[str, Any]:
    updated_record = dict(record)
    if updated_record.get("status") != "ok" or not is_vci_cot_prompt_mode(prompt_mode):
        return updated_record

    structured_output = extract_vci_cot_steps(str(updated_record.get("raw_output", "")), prompt_mode)
    for step_number in range(1, 5):
        updated_record.pop(f"vci_cot_step_{step_number}", None)
    updated_record.update(
        {
            key: value
            for key, value in structured_output.items()
            if key.startswith("vci_cot_step_")
        }
    )
    updated_record["vci_cot_has_complete_steps"] = bool(structured_output.get("vci_cot_has_complete_steps"))
    return updated_record


def parse_model_output(
    text: str,
    sample: VciSample,
    prompt_mode: str = "answer_only_extractable",
) -> Tuple[Optional[int], str, Dict[str, Any]]:
    structured_output: Dict[str, Any] = {}
    candidate_text = extract_final_answer_region(text)
    if is_vci_cot_prompt_mode(prompt_mode):
        structured_output = extract_vci_cot_steps(text, prompt_mode)
        candidate_text = structured_output.get("vci_cot_step_4") or candidate_text

    pred_idx, prediction_source = parse_prediction(candidate_text, sample)
    return pred_idx, prediction_source, structured_output


def build_content_blocks(prompt_text: str, sample: VciSample, input_modality: str = "multimodal") -> List[Dict[str, str]]:
    blocks = [{"type": "text", "text": prompt_text}]
    if input_modality != "text_only":
        blocks.append({"type": "image_path", "image_path": str(sample.image_path)})
    return blocks


def run_single_inference(client: InferenceAPIClient, prompt_text: str, system_message: str, sample: VciSample, args: Any) -> str:
    last_error: Optional[APIClientError] = None
    input_modality = getattr(args, "input_modality", "multimodal")
    content_blocks = build_content_blocks(prompt_text, sample, input_modality=input_modality)
    image_path = str(sample.image_path) if input_modality != "text_only" else None
    for attempt in range(args.max_retries + 1):
        try:
            return client.generate(
                prompt_text=prompt_text,
                system_message=system_message,
                image_path=image_path,
                content_blocks=content_blocks,
            )
        except APIClientError as exc:
            last_error = exc
            if attempt >= args.max_retries:
                break
            retry_count = attempt + 1
            total_attempts = args.max_retries + 1
            print(
                f"[retry {retry_count}/{args.max_retries}] "
                f"qid={sample.qid} next_attempt={retry_count + 1}/{total_attempts} error={exc}",
                flush=True,
            )
    if last_error is None:
        raise RuntimeError("Inference failed without a captured exception.")
    raise last_error


def record_from_success(
    sample: VciSample,
    output_text: str,
    pred_idx: int,
    prediction_source: str,
    structured_output: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    pred_label = sample.option_labels[pred_idx]
    record = {
        "sample_key": sample.sample_key,
        "qid": sample.qid,
        "domain": sample.domain,
        "question_scope": sample.question_scope,
        "mechanism_id": sample.mechanism_id,
        "variant_id": sample.variant_id,
        "transfer_type": sample.transfer_type,
        "image_path": str(sample.image_path),
        "status": "ok",
        "gold_index": sample.answer_idx,
        "gold_label": sample.answer_label,
        "gold_choice": sample.answer_text,
        "pred_index": pred_idx,
        "pred_label": pred_label,
        "pred_choice": sample.choices[pred_idx],
        "prediction_source": prediction_source,
        "is_correct": pred_idx == sample.answer_idx,
        "answer_prior": sample.answer_prior,
        "matches_answer_prior": bool(sample.answer_prior) and pred_label == sample.answer_prior,
        "raw_output": output_text,
    }
    if structured_output:
        record.update(structured_output)
    return record


def record_from_error(sample: VciSample, status: str, error_message: str, raw_output: str = "") -> Dict[str, Any]:
    return {
        "sample_key": sample.sample_key,
        "qid": sample.qid,
        "domain": sample.domain,
        "question_scope": sample.question_scope,
        "mechanism_id": sample.mechanism_id,
        "variant_id": sample.variant_id,
        "transfer_type": sample.transfer_type,
        "image_path": str(sample.image_path),
        "status": status,
        "error_message": error_message,
        "is_correct": False,
        "matches_answer_prior": False,
        "raw_output": raw_output,
    }


def get_record_sample_key(record: Dict[str, Any]) -> str:
    sample_key = str(record.get("sample_key", "")).strip()
    if sample_key:
        return sample_key

    domain = str(record.get("domain", "")).strip()
    qid = str(record.get("qid", "")).strip()
    if domain and qid:
        return build_sample_key(domain, qid)
    return qid


def load_existing_records(predictions_path: Path) -> Dict[str, Dict[str, Any]]:
    latest_records: Dict[str, Dict[str, Any]] = {}
    for record in load_jsonl(predictions_path):
        latest_records[get_record_sample_key(record)] = record
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
        output_dir = Path(args.output_root) / safe_model / args.label
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def print_dry_run_preview(client: InferenceAPIClient, sample: VciSample, prompt_text: str, system_message: str) -> None:
    input_modality = getattr(client, "_vci_input_modality", "multimodal")
    content_blocks = build_content_blocks(prompt_text, sample, input_modality=input_modality)
    image_path = str(sample.image_path) if input_modality != "text_only" else None
    payload = client.build_payload(
        prompt_text=prompt_text,
        system_message=system_message,
        image_path=image_path,
        content_blocks=content_blocks,
    )
    print("====Dry Run Prompt====")
    print(prompt_text)
    print("")
    print("====Dry Run System Message====")
    print(system_message)
    print("")
    print("====Dry Run Payload (redacted)====")
    print(json.dumps(redact_payload_for_preview(payload), indent=2, ensure_ascii=False))


def run_vci_evaluation(args: Any) -> Dict[str, Any]:
    random.seed(args.seed)
    output_dir = ensure_output_dir(args)
    exclusion_spec = load_sample_exclusion_spec(args.exclude_json) if getattr(args, "exclude_json", None) else None

    samples, preflight_report = load_vci_samples(
        data_root=args.data_root,
        question_subdir=getattr(args, "question_subdir", "shuffled"),
        domains=args.domain,
        question_scopes=args.question_scope,
        transfer_types=args.transfer_type,
        qids=args.qids,
        sample_keys=getattr(args, "sample_keys", None),
        exclusion_spec=exclusion_spec,
        limit=args.limit,
    )
    save_json(output_dir / "preflight_report.json", preflight_report)
    if not samples:
        raise ValueError("No valid samples matched the current filters.")

    client = create_client(args)
    client._vci_input_modality = args.input_modality
    prompt_text, system_message = build_prompt(
        samples[0],
        prompt_mode=args.prompt_mode,
        input_modality=args.input_modality,
        text_context_source=args.text_context_source,
    )
    if args.dry_run:
        print_dry_run_preview(client, samples[0], prompt_text, system_message)
        return {
            "dry_run": True,
            "sample_qid": samples[0].qid,
            "output_dir": str(output_dir),
        }

    predictions_path = output_dir / "predictions.jsonl"
    existing_records = load_existing_records(predictions_path) if args.resume else {}
    records_by_sample_key: Dict[str, Dict[str, Any]] = dict(existing_records)
    completed_sample_keys = {
        sample_key
        for sample_key, record in existing_records.items()
        if str(record.get("status")) in {"ok", "parse_error"}
    }

    for index, sample in enumerate(samples, start=1):
        if sample.sample_key in completed_sample_keys:
            continue

        prompt_text, system_message = build_prompt(
            sample,
            prompt_mode=args.prompt_mode,
            input_modality=args.input_modality,
            text_context_source=args.text_context_source,
        )
        try:
            output_text = run_single_inference(client, prompt_text, system_message, sample, args)
        except APIClientError as exc:
            record = record_from_error(sample, status="api_error", error_message=str(exc))
        else:
            pred_idx, prediction_source, structured_output = parse_model_output(
                output_text,
                sample,
                prompt_mode=args.prompt_mode,
            )
            if pred_idx is None:
                record = record_from_error(
                    sample,
                    status="parse_error",
                    error_message="Unable to map model output to an option.",
                    raw_output=output_text,
                )
            else:
                record = record_from_success(
                    sample,
                    output_text,
                    pred_idx,
                    prediction_source,
                    structured_output=structured_output,
                )

        append_jsonl(predictions_path, [record])
        records_by_sample_key[sample.sample_key] = record
        if record["status"] == "ok":
            completed_sample_keys.add(sample.sample_key)

        if args.debug or index <= 3:
            print(f"[{index}/{len(samples)}] qid={sample.qid} status={record['status']} correct={record.get('is_correct')}")

    ordered_records = [
        records_by_sample_key[sample.sample_key]
        for sample in samples
        if sample.sample_key in records_by_sample_key
    ]
    run_config = sanitize_args_for_logging(args)
    run_config["output_dir"] = str(output_dir)
    report_bundle = write_report_bundle(output_dir, ordered_records, preflight_report, run_config)
    return {
        "output_dir": str(output_dir),
        "summary": report_bundle["summary"],
    }
