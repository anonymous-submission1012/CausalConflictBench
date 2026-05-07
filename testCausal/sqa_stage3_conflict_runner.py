from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from models.inference_api import APIClientError, InferenceAPIClient
from testCausal.sqa_stage3_conflict_dataset import (
    Stage3ConflictSample,
    append_jsonl,
    load_jsonl,
    load_stage3_conflict_samples,
    save_json,
)
from testCausal.sqa_stage3_conflict_prompt import (
    build_prompt,
    get_ca_cot_required_steps,
    is_ca_cot_prompt_mode,
)
from testCausal.sqa_stage3_conflict_report import write_report_bundle

RESUME_COMPLETED_STATUSES = {"ok", "parse_error"}


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
            "(?:^| )(?:answer|final answer|correct answer|the answer is|the correct answer is|option|choice|\\u7b54\\u6848)\\s*[:\\uff1a]?\\s*(.+)$",
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


def parse_prediction(text: str, sample: Stage3ConflictSample) -> Tuple[Optional[int], str]:
    option_letter = extract_option_letter(text, sample.option_labels)
    if option_letter is not None:
        return sample.option_labels.index(option_letter), "option_letter"

    choice_idx = extract_choice_text_match(text, sample.choices)
    if choice_idx is not None:
        return choice_idx, "choice_text"

    return None, "unparsed"


def extract_ca_cot_steps(text: str, prompt_mode: str) -> Dict[str, Any]:
    normalized_text = text.replace("\r\n", "\n").strip()
    step_label_pattern = (
        r"(?:"
        r"visual\s*observation|"
        r"conflict\s*detection|"
        r"rule[-\s]*conditioned\s*reasoning|"
        r"rule\s*recognition|"
        r"rulerecognition|"
        r"reasoning|"
        r"final\s*answer|"
        r"finalanswer"
        r")"
    )
    matches = list(
        re.finditer(
            rf"(?i)\bstep\s*([1-4])\s*:\s*(?:{step_label_pattern})?",
            normalized_text,
        )
    )
    step_payload: Dict[str, Any] = {}
    if not matches:
        step_payload["ca_cot_has_complete_steps"] = False
        return step_payload

    extracted_steps: Dict[str, str] = {}
    for index, match in enumerate(matches):
        step_number = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized_text)
        content = normalized_text[start:end].strip()
        content = re.sub(r"^\s*[-–—:：]\s*", "", content)
        extracted_steps[f"ca_cot_step_{step_number}"] = content

    step_payload.update(extracted_steps)
    required_steps = get_ca_cot_required_steps(prompt_mode)
    step_payload["ca_cot_has_complete_steps"] = all(
        bool(step_payload.get(f"ca_cot_step_{step_number}")) for step_number in required_steps
    )
    return step_payload


def parse_model_output(
    text: str,
    sample: Stage3ConflictSample,
    prompt_mode: str = "answer_only_extractable",
) -> Tuple[Optional[int], str, Dict[str, Any]]:
    structured_output: Dict[str, Any] = {}
    candidate_text = extract_final_answer_region(text)
    if is_ca_cot_prompt_mode(prompt_mode):
        structured_output = extract_ca_cot_steps(text, prompt_mode)
        final_step_number = 2 if prompt_mode == "ca_cot_zero_shot_2step" else 4
        candidate_text = structured_output.get(f"ca_cot_step_{final_step_number}") or candidate_text

    pred_idx, prediction_source = parse_prediction(candidate_text, sample)
    return pred_idx, prediction_source, structured_output


def build_content_blocks(prompt_text: str, sample: Stage3ConflictSample, input_modality: str = "multimodal") -> List[Dict[str, str]]:
    blocks: List[Dict[str, str]] = [{"type": "text", "text": prompt_text}]
    if input_modality == "multimodal":
        blocks.append({"type": "image_path", "image_path": str(sample.image_path)})
    return blocks


def run_single_inference(client: InferenceAPIClient, prompt_text: str, system_message: str, sample: Stage3ConflictSample, args: Any) -> str:
    last_error: Optional[APIClientError] = None
    content_blocks = build_content_blocks(prompt_text, sample, input_modality=args.input_modality)
    image_path = str(sample.image_path) if args.input_modality == "multimodal" else None
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
                f"sample_id={sample.sample_id} next_attempt={retry_count + 1}/{total_attempts} error={exc}",
                flush=True,
            )
    if last_error is None:
        raise RuntimeError("Inference failed without a captured exception.")
    raise last_error


def record_from_success(
    sample: Stage3ConflictSample,
    output_text: str,
    pred_idx: int,
    prediction_source: str,
    structured_output: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    pred_label = sample.option_labels[pred_idx]
    gold_label = sample.option_labels[sample.answer_idx]
    matches_factual_baseline = None
    if sample.task_variant == "conflict" and sample.factual_baseline_index is not None:
        matches_factual_baseline = pred_idx == sample.factual_baseline_index
    record = {
        "sample_id": sample.sample_id,
        "qid": sample.qid,
        "flip_index": sample.flip_index,
        "task_variant": sample.task_variant,
        "split": sample.split,
        "topic": sample.topic,
        "category": sample.category,
        "subject": sample.subject,
        "causal_type": sample.causal_type,
        "conflict_intensity": sample.conflict_intensity,
        "reasoning_complexity": sample.reasoning_complexity,
        "donor_qid": sample.donor_qid,
        "donor_topic": sample.donor_topic,
        "donor_category": sample.donor_category,
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
        "factual_baseline_index": sample.factual_baseline_index,
        "factual_baseline_choice": sample.factual_baseline_text,
        "matches_factual_baseline": matches_factual_baseline,
        "raw_output": output_text,
    }
    if structured_output:
        record.update(structured_output)
    return record


def record_from_error(
    sample: Stage3ConflictSample,
    status: str,
    error_message: str,
    raw_output: str = "",
    structured_output: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    record = {
        "sample_id": sample.sample_id,
        "qid": sample.qid,
        "flip_index": sample.flip_index,
        "task_variant": sample.task_variant,
        "split": sample.split,
        "topic": sample.topic,
        "category": sample.category,
        "subject": sample.subject,
        "causal_type": sample.causal_type,
        "conflict_intensity": sample.conflict_intensity,
        "reasoning_complexity": sample.reasoning_complexity,
        "donor_qid": sample.donor_qid,
        "donor_topic": sample.donor_topic,
        "donor_category": sample.donor_category,
        "image_path": str(sample.image_path),
        "status": status,
        "error_message": error_message,
        "is_correct": False,
        "factual_baseline_index": sample.factual_baseline_index,
        "factual_baseline_choice": sample.factual_baseline_text,
        "matches_factual_baseline": False if sample.task_variant == "conflict" else None,
        "raw_output": raw_output,
    }
    if structured_output:
        record.update(structured_output)
    return record


def load_existing_records(predictions_path: Path) -> Dict[str, Dict[str, Any]]:
    latest_records: Dict[str, Dict[str, Any]] = {}
    for record in load_jsonl(predictions_path):
        latest_records[str(record["sample_id"])] = record
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
        output_dir = Path(args.output_root) / "stage3_conflict_eval" / safe_model / args.label
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def persist_ca_cot_artifacts(output_dir: Path | str, prompt_text: str, system_message: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    preview_text = "\n\n".join(
        [
            "====System Message====",
            system_message.strip(),
            "====Prompt Preview====",
            prompt_text.strip(),
        ]
    ).strip() + "\n"
    (output_path / "ca_cot_prompt_preview.txt").write_text(preview_text, encoding="utf-8")


def print_dry_run_preview(client: InferenceAPIClient, sample: Stage3ConflictSample, prompt_text: str, system_message: str, args: Any) -> None:
    content_blocks = build_content_blocks(prompt_text, sample, input_modality=args.input_modality)
    image_path = str(sample.image_path) if args.input_modality == "multimodal" else None
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


def run_stage3_conflict_evaluation(args: Any) -> Dict[str, Any]:
    random.seed(args.seed)
    output_dir = ensure_output_dir(args)

    samples, preflight_report = load_stage3_conflict_samples(
        data_file=args.data_file,
        image_root=args.image_root,
        captions_file=args.captions_file,
        split=None if args.split == "all" else args.split,
        limit=args.limit,
        topics=args.topic,
        categories=args.category,
        causal_types=args.causal_type,
        conflict_intensities=args.conflict_intensity,
        qids=args.qids,
        sample_ids=args.sample_ids,
        task_variant=args.task_variant,
        distractor_rule_intensities=args.distractor_rule_intensity,
    )
    save_json(output_dir / "preflight_report.json", preflight_report)

    if not samples:
        raise ValueError("No valid conflict samples matched the current filters.")

    client = create_client(args)
    prompt_text, system_message = build_prompt(
        samples[0],
        prompt_mode=args.prompt_mode,
        rule_position=args.rule_position,
        input_modality=args.input_modality,
        text_context_source=args.text_context_source,
    )
    if is_ca_cot_prompt_mode(args.prompt_mode):
        persist_ca_cot_artifacts(output_dir, prompt_text, system_message)
    if args.dry_run:
        print_dry_run_preview(client, samples[0], prompt_text, system_message, args)
        return {
            "dry_run": True,
            "sample_id": samples[0].sample_id,
            "output_dir": str(output_dir),
        }

    predictions_path = output_dir / "predictions.jsonl"
    existing_records = load_existing_records(predictions_path) if args.resume else {}
    records_by_sample_id: Dict[str, Dict[str, Any]] = dict(existing_records)
    completed_sample_ids = {
        sample_id
        for sample_id, record in existing_records.items()
        if str(record.get("status")) in RESUME_COMPLETED_STATUSES
    }

    for index, sample in enumerate(samples, start=1):
        if sample.sample_id in completed_sample_ids:
            continue

        prompt_text, system_message = build_prompt(
            sample,
            prompt_mode=args.prompt_mode,
            rule_position=args.rule_position,
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
                    structured_output=structured_output,
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
        records_by_sample_id[sample.sample_id] = record
        if record["status"] == "ok":
            completed_sample_ids.add(sample.sample_id)

        if args.debug or index <= 3:
            print(f"[{index}/{len(samples)}] sample_id={sample.sample_id} status={record['status']} correct={record.get('is_correct')}")

    run_config = sanitize_args_for_logging(args)
    run_config["output_dir"] = str(output_dir)
    ordered_records = [
        records_by_sample_id[sample.sample_id]
        for sample in samples
        if sample.sample_id in records_by_sample_id
    ]
    report_bundle = write_report_bundle(output_dir, ordered_records, preflight_report, run_config)
    return {
        "output_dir": str(output_dir),
        "summary": report_bundle["summary"],
    }
