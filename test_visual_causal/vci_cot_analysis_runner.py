from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from models.inference_api import APIClientError, InferenceAPIClient
from test_visual_causal.vci_cot_analysis_prompt import FLIP_RULE_LABELS, RULE_SOURCE_LABELS, build_prompt
from test_visual_causal.vci_cot_analysis_report import write_report_bundle
from test_visual_causal.vci_eval_dataset import append_jsonl, load_json, load_jsonl
from test_visual_causal.vci_eval_runner import redact_payload_for_preview, resolve_api_key


def create_client(args: Any) -> InferenceAPIClient:
    api_key, api_key_source = resolve_api_key(args.api_key, args.api_key_env, allow_dummy=args.dry_run)
    args.api_key_source = api_key_source
    return InferenceAPIClient(
        base_url=args.api_base_url,
        endpoint_type=args.endpoint_type,
        model=args.judge_model,
        api_key=api_key,
        timeout=args.timeout,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_tokens,
        stream=args.stream,
        store=args.store,
        reasoning_effort=args.reasoning_effort,
        user_agent=args.user_agent,
    )


def _extract_label_from_text(text: str, candidates: Sequence[str]) -> str:
    lowered = text.lower()
    for candidate in sorted(candidates, key=len, reverse=True):
        pattern = rf"(?<![a-z_]){re.escape(candidate.lower())}(?![a-z_])"
        if re.search(pattern, lowered):
            return candidate
    return ""


def parse_judge_output(text: str, analysis_mode: str = "flip_rule") -> Dict[str, str]:
    normalized = text.strip()
    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError:
        payload = None

    expected_fields: list[tuple[str, Sequence[str]]]
    if analysis_mode == "rule_source":
        expected_fields = [("cot_rule_source_label", sorted(RULE_SOURCE_LABELS))]
    elif analysis_mode == "both":
        expected_fields = [
            ("cot_rule_label", sorted(FLIP_RULE_LABELS)),
            ("cot_rule_source_label", sorted(RULE_SOURCE_LABELS)),
        ]
    else:
        expected_fields = [("cot_rule_label", sorted(FLIP_RULE_LABELS))]

    parsed: Dict[str, str] = {}
    for field_name, candidates in expected_fields:
        label_value = ""
        if isinstance(payload, dict):
            label_value = str(payload.get(field_name, "")).strip().lower()
        if not label_value and normalized:
            label_value = _extract_label_from_text(normalized, candidates)
        if label_value not in candidates:
            raise ValueError(f"Unsupported {field_name}: {label_value or normalized}")
        parsed[field_name] = label_value
    return parsed


def derive_joint_bucket(cot_rule_label: str, is_correct: bool) -> str:
    if cot_rule_label == "correct":
        return "cot_correct_answer_correct" if is_correct else "cot_correct_answer_wrong"
    if cot_rule_label == "incorrect":
        return "cot_wrong_answer_correct" if is_correct else "cot_wrong_answer_wrong"
    return "cot_undecidable"


def load_run_context(args: Any) -> Dict[str, Any]:
    run_dir = Path(args.run_dir) if getattr(args, "run_dir", None) else None
    predictions_path = Path(args.predictions_path) if getattr(args, "predictions_path", None) else None

    if predictions_path is None:
        if run_dir is None:
            raise ValueError("Either --predictions_path or --run_dir is required.")
        predictions_path = run_dir / "predictions.jsonl"
    if run_dir is None:
        run_dir = predictions_path.parent

    run_config_path = run_dir / "run_config.json"
    run_config = load_json(run_config_path) if run_config_path.exists() else {}
    return {
        "run_dir": run_dir,
        "predictions_path": predictions_path,
        "predictions": load_jsonl(predictions_path),
        "run_config": run_config,
    }


def resolve_output_dir(run_dir: Path, args: Any, create: bool = True) -> Path:
    if getattr(args, "output_dir", None):
        output_dir = Path(args.output_dir)
    else:
        label = getattr(args, "judge_label", None) or f"cot_analysis_{args.judge_model}"
        output_dir = run_dir / label
    if create:
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_reference_index(data_root: Path | str, question_subdir: str) -> Dict[str, List[Dict[str, Any]]]:
    question_root = Path(data_root) / question_subdir
    references: Dict[str, List[Dict[str, Any]]] = {}
    for question_file in sorted(question_root.glob("*.json")):
        rows = load_json(question_file)
        if not isinstance(rows, list):
            continue
        for row in rows:
            qid = str(row.get("question_id", "")).strip()
            if not qid:
                continue
            references.setdefault(qid, []).append(dict(row))
    return references


def resolve_reference_row(
    prediction_row: Dict[str, Any],
    references: Dict[str, List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    qid = str(prediction_row.get("qid", "")).strip()
    candidates = references.get(qid, [])
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    match_fields = (
        "domain",
        "question_scope",
        "mechanism_id",
        "variant_id",
        "transfer_type",
    )
    exact_matches = [
        row
        for row in candidates
        if all(
            str(row.get(field, "")).strip() == str(prediction_row.get(field, "")).strip()
            for field in match_fields
        )
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    return None


def filter_predictions(rows: Sequence[Dict[str, Any]], qids: Optional[Sequence[str]], limit: Optional[int]) -> List[Dict[str, Any]]:
    allowed_qids = {qid.strip() for qid in qids or [] if qid and qid.strip()}
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        qid = str(row.get("qid", "")).strip()
        if allowed_qids and qid not in allowed_qids:
            continue
        filtered.append(row)
        if limit is not None and len(filtered) >= limit:
            break
    return filtered


def print_dry_run_preview(client: InferenceAPIClient, prompt_text: str, system_message: str) -> None:
    payload = client.build_payload(
        prompt_text=prompt_text,
        system_message=system_message,
        content_blocks=[{"type": "text", "text": prompt_text}],
    )
    print("====Dry Run Prompt====")
    print(prompt_text)
    print("")
    print("====Dry Run System Message====")
    print(system_message)
    print("")
    print("====Dry Run Payload (redacted)====")
    print(json.dumps(redact_payload_for_preview(payload), indent=2, ensure_ascii=False))


def run_single_judge(client: InferenceAPIClient, prompt_text: str, system_message: str, args: Any) -> str:
    last_error: Optional[APIClientError] = None
    for attempt in range(args.max_retries + 1):
        try:
            return client.generate(
                prompt_text=prompt_text,
                system_message=system_message,
                content_blocks=[{"type": "text", "text": prompt_text}],
            )
        except APIClientError as exc:
            last_error = exc
            if attempt >= args.max_retries:
                break
    if last_error is None:
        raise RuntimeError("Judge call failed without a captured exception.")
    raise last_error


def build_base_record(prediction_row: Dict[str, Any], reference_row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    record = dict(prediction_row)
    if reference_row:
        record["required_induction"] = str(reference_row.get("required_induction", "")).strip()
        record["flipped_rule"] = str(reference_row.get("flipped_rule", "")).strip()
        record["question"] = str(reference_row.get("question", "")).strip()
        record["choices"] = dict(reference_row.get("choices", {})) if isinstance(reference_row.get("choices"), dict) else {}
    return record


def build_prediction_record_key(row: Dict[str, Any]) -> Tuple[str, str, str, str, str, str]:
    return (
        str(row.get("qid", "")).strip(),
        str(row.get("domain", "")).strip(),
        str(row.get("question_scope", "")).strip(),
        str(row.get("mechanism_id", "")).strip(),
        str(row.get("variant_id", "")).strip(),
        str(row.get("transfer_type", "")).strip(),
    )


def run_vci_cot_analysis(args: Any) -> Dict[str, Any]:
    random.seed(args.seed)
    context = load_run_context(args)
    run_dir = context["run_dir"]
    predictions = filter_predictions(context["predictions"], getattr(args, "qids", None), getattr(args, "limit", None))
    if not predictions:
        raise ValueError("No predictions matched the requested filters.")

    question_subdir = getattr(args, "question_subdir", None) or context["run_config"].get("question_subdir") or "shuffled"
    references = load_reference_index(args.data_root, question_subdir)
    output_dir = resolve_output_dir(run_dir, args, create=not args.dry_run)
    eligible_item: Optional[Tuple[Dict[str, Any], Dict[str, Any], str, str]] = None
    for prediction_row in predictions:
        reference_row = resolve_reference_row(prediction_row, references)
        if not reference_row:
            continue
        if str(prediction_row.get("status", "")) != "ok":
            continue
        if not str(prediction_row.get("vci_cot_step_1", "")).strip():
            continue
        prompt_text, system_message = build_prompt(
            prediction_row,
            reference_row,
            analysis_mode=getattr(args, "analysis_mode", "flip_rule"),
        )
        eligible_item = (prediction_row, reference_row, prompt_text, system_message)
        break

    if eligible_item is None:
        raise ValueError("No eligible predictions contained a usable vci_cot_step_1 for analysis.")

    if args.dry_run:
        client = create_client(args)
        prediction_row, _, prompt_text, system_message = eligible_item
        print_dry_run_preview(client, prompt_text, system_message)
        return {
            "dry_run": True,
            "sample_qid": prediction_row["qid"],
            "output_dir": str(output_dir),
        }

    output_path = output_dir / "cot_analysis.jsonl"
    if not args.resume and output_path.exists():
        output_path.unlink()
    existing_records = {build_prediction_record_key(row): row for row in load_jsonl(output_path)} if args.resume and output_path.exists() else {}
    completed_record_keys = {
        record_key
        for record_key, row in existing_records.items()
        if str(row.get("analysis_status", "")) in {"ok", "missing_reference", "prediction_not_ok", "missing_step1"}
    }

    records_by_key: Dict[Tuple[str, str, str, str, str, str], Dict[str, Any]] = {}

    def build_analysis_record(
        prediction_row: Dict[str, Any],
        client: InferenceAPIClient,
    ) -> Tuple[Tuple[str, str, str, str, str, str], Dict[str, Any], bool]:
        record_key = build_prediction_record_key(prediction_row)
        if record_key in completed_record_keys:
            return record_key, existing_records[record_key], False

        reference_row = resolve_reference_row(prediction_row, references)
        base_record = build_base_record(prediction_row, reference_row)
        if not reference_row:
            base_record["analysis_status"] = "missing_reference"
            return record_key, base_record, True
        if str(prediction_row.get("status", "")) != "ok":
            base_record["analysis_status"] = "prediction_not_ok"
            return record_key, base_record, True
        if not str(prediction_row.get("vci_cot_step_1", "")).strip():
            base_record["analysis_status"] = "missing_step1"
            return record_key, base_record, True

        prompt_text, system_message = build_prompt(
            prediction_row,
            reference_row,
            analysis_mode=getattr(args, "analysis_mode", "flip_rule"),
        )
        judge_output = ""
        try:
            judge_output = run_single_judge(client, prompt_text, system_message, args)
            parsed = parse_judge_output(judge_output, analysis_mode=getattr(args, "analysis_mode", "flip_rule"))
        except (APIClientError, ValueError) as exc:
            base_record["analysis_status"] = "judge_error"
            base_record["judge_raw_output"] = judge_output
            base_record["judge_error"] = str(exc)
            return record_key, base_record, True

        base_record["analysis_status"] = "ok"
        base_record["judge_raw_output"] = judge_output
        base_record.update(parsed)
        cot_rule_label = parsed.get("cot_rule_label")
        if cot_rule_label:
            base_record["joint_bucket"] = derive_joint_bucket(cot_rule_label, bool(prediction_row.get("is_correct")))
        return record_key, base_record, True

    def persist_if_needed(record: Dict[str, Any], should_persist: bool) -> None:
        if should_persist:
            append_jsonl(output_path, [record])

    worker_count = max(1, int(getattr(args, "workers", 1) or 1))
    if worker_count == 1:
        client = create_client(args)
        for prediction_row in predictions:
            record_key, record, should_persist = build_analysis_record(prediction_row, client)
            persist_if_needed(record, should_persist)
            records_by_key[record_key] = record
    else:
        thread_local = threading.local()

        def get_thread_client() -> InferenceAPIClient:
            client = getattr(thread_local, "client", None)
            if client is None:
                client = create_client(args)
                thread_local.client = client
            return client

        def build_analysis_record_in_thread(
            prediction_row: Dict[str, Any],
        ) -> Tuple[Tuple[str, str, str, str, str, str], Dict[str, Any], bool]:
            return build_analysis_record(prediction_row, get_thread_client())

        max_workers = min(worker_count, len(predictions))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_prediction = {
                executor.submit(build_analysis_record_in_thread, prediction_row): prediction_row
                for prediction_row in predictions
            }
            for future in as_completed(future_to_prediction):
                record_key, record, should_persist = future.result()
                persist_if_needed(record, should_persist)
                records_by_key[record_key] = record

    records = [
        records_by_key[build_prediction_record_key(prediction_row)]
        for prediction_row in predictions
        if build_prediction_record_key(prediction_row) in records_by_key
    ]

    judge_config = {
        "judge_model": args.judge_model,
        "endpoint_type": args.endpoint_type,
        "api_base_url": args.api_base_url,
    }
    run_config = {
        "source_predictions_path": str(context["predictions_path"]),
        "source_run_dir": str(run_dir),
        "question_subdir": question_subdir,
        "judge_label": getattr(args, "judge_label", None) or f"cot_analysis_{args.judge_model}",
        "analysis_mode": getattr(args, "analysis_mode", "flip_rule"),
    }
    bundle = write_report_bundle(output_dir=output_dir, records=records, judge_config=judge_config, run_config=run_config)
    return {
        "output_dir": str(output_dir),
        "summary": bundle["summary"],
    }
