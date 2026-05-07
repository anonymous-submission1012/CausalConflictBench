from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.inference_api import APIClientError, InferenceAPIClient
from testCausal.sqa_stage3_conflict_dataset import (
    Stage3ConflictSample,
    load_jsonl,
    load_stage3_conflict_samples,
)
from testCausal.sqa_stage3_conflict_prompt import is_ca_cot_prompt_mode
from testCausal.sqa_stage3_conflict_runner import extract_ca_cot_steps
from testCausal.sqa_stage3_conflict_ca_cot_report import write_ca_cot_judge_report_bundle


SYSTEM_MESSAGE = "You are a strict JSON judge for CA-CoT rule recognition outputs. Return JSON only."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Judge CA-CoT predictions from an existing Stage 3 run directory.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--judge_model", type=str, default="gpt-5.4")
    parser.add_argument("--judge_api_base_url", type=str, default="<LOCAL_API_BASE_URL_PLACEHOLDER>")
    parser.add_argument(
        "--judge_endpoint_type",
        type=str,
        default="responses",
        choices=["responses", "chat_completions"],
    )
    parser.add_argument("--judge_api_key", type=str, default=None)
    parser.add_argument("--judge_api_key_env", type=str, default="INFINITEAI_API_KEY")
    parser.add_argument("--judge_reasoning_effort", type=str, default="medium")
    parser.add_argument("--judge_max_tokens", type=int, default=800)
    parser.add_argument("--judge_timeout", type=int, default=180)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser


def resolve_judge_output_dir(run_dir: Path | str, judge_model: str) -> Path:
    return Path(run_dir) / f"ca_cot_judge_{judge_model}"


def _normalize_project_input_path(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return value

    path = Path(text)
    normalized = text.replace("\\", "/")
    is_absolute_like = path.is_absolute() or normalized.startswith("/")
    if path.exists() or not is_absolute_like:
        return value

    for project_dir in ("data",):
        marker = f"/{project_dir}/"
        marker_index = normalized.find(marker)
        if marker_index != -1:
            relative = normalized[marker_index + 1 :]
            if (PROJECT_ROOT / Path(relative)).exists():
                return relative
    return value


def _normalize_run_config_paths(run_config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(run_config)
    for key in ("data_file", "image_root", "captions_file"):
        normalized[key] = _normalize_project_input_path(normalized.get(key))
    return normalized


def load_run_context(run_dir: Path | str) -> Dict[str, Any]:
    run_path = Path(run_dir)
    run_config = json.loads((run_path / "run_config.json").read_text(encoding="utf-8-sig"))
    predictions = load_jsonl(run_path / "predictions.jsonl")
    return {
        "run_dir": run_path,
        "run_config": _normalize_run_config_paths(run_config),
        "predictions": predictions,
    }


def derive_quadrant(cd_score: float, is_correct: bool) -> str:
    if cd_score >= 0.5:
        return "A" if is_correct else "B"
    return "C" if is_correct else "D"


def derive_error_type(is_correct: bool, cd_score: float, matches_factual_baseline: Optional[bool]) -> str:
    if is_correct:
        return "none"
    if cd_score >= 0.5:
        return "type_ii"
    if matches_factual_baseline is True:
        return "type_i"
    if cd_score < 0.5:
        return "undetermined"
    return "undetermined"


def _resolve_api_key(cli_value: Optional[str], env_name: Optional[str]) -> str:
    if cli_value:
        return cli_value
    if env_name:
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    raise ValueError("Judge API key not found. Use --judge_api_key or set the configured environment variable.")


def _build_client(args: argparse.Namespace) -> InferenceAPIClient:
    return InferenceAPIClient(
        base_url=args.judge_api_base_url,
        endpoint_type=args.judge_endpoint_type,
        model=args.judge_model,
        api_key=_resolve_api_key(args.judge_api_key, args.judge_api_key_env),
        timeout=args.judge_timeout,
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=args.judge_max_tokens,
        stream=False,
        store=False,
        reasoning_effort=args.judge_reasoning_effort,
    )


def _resolve_preview_api_key(cli_value: Optional[str], env_name: Optional[str]) -> str:
    if cli_value:
        return cli_value
    if env_name:
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return "<dry-run-api-key>"


def _build_preview_client(args: argparse.Namespace) -> InferenceAPIClient:
    return InferenceAPIClient(
        base_url=args.judge_api_base_url,
        endpoint_type=args.judge_endpoint_type,
        model=args.judge_model,
        api_key=_resolve_preview_api_key(args.judge_api_key, args.judge_api_key_env),
        timeout=args.judge_timeout,
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=args.judge_max_tokens,
        stream=False,
        store=False,
        reasoning_effort=args.judge_reasoning_effort,
    )


def _redact_payload_for_preview(payload: Any) -> Any:
    if isinstance(payload, dict):
        redacted: Dict[str, Any] = {}
        for key, value in payload.items():
            if key in {"api_key", "Authorization", "authorization"}:
                redacted[key] = "***"
            elif key in {"image_url", "url"} and isinstance(value, str) and value.startswith("data:"):
                redacted[key] = "<data-url omitted>"
            else:
                redacted[key] = _redact_payload_for_preview(value)
        return redacted
    if isinstance(payload, list):
        return [_redact_payload_for_preview(item) for item in payload]
    return payload


def _redact_headers_for_preview(headers: Dict[str, str]) -> Dict[str, str]:
    redacted = dict(headers)
    if "Authorization" in redacted:
        redacted["Authorization"] = "Bearer ***"
    return redacted


def _extract_json_object(raw_output: str) -> Dict[str, Any]:
    text = raw_output.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Judge output does not contain a valid JSON object.")
        return json.loads(text[start : end + 1])


def _normalize_score(value: Any, allowed: set[float], field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc
    if parsed not in allowed:
        raise ValueError(f"{field_name} has unsupported value: {parsed}")
    return parsed


def _parse_judge_output(raw_output: str) -> Dict[str, Any]:
    payload = _extract_json_object(raw_output)
    return {"cd_score": _normalize_score(payload.get("cd_score"), {0.0, 0.5, 1.0}, "cd_score")}


def _select_rule_recognition_step(prompt_mode: str, prediction: Dict[str, Any]) -> Tuple[str, str]:
    if prompt_mode == "ca_cot_zero_shot_2step":
        field_name = "ca_cot_step_1"
    else:
        field_name = "ca_cot_step_2"
    return field_name, str(prediction.get(field_name, "")).strip()


def _with_reparsed_ca_cot_steps(prediction: Dict[str, Any], prompt_mode: str) -> Dict[str, Any]:
    if prediction.get("ca_cot_has_complete_steps", True):
        return prediction

    reparsed_steps = extract_ca_cot_steps(str(prediction.get("raw_output") or ""), prompt_mode)
    if reparsed_steps.get("ca_cot_has_complete_steps"):
        return {**prediction, **reparsed_steps}
    return prediction


def _build_judge_prompt(sample: Stage3ConflictSample, prediction: Dict[str, Any], prompt_mode: str) -> str:
    _, rule_recognition_text = _select_rule_recognition_step(prompt_mode, prediction)
    original_rule = str(sample.source_record.get("problem", {}).get("rule", "")).strip()
    return f"""You are judging whether a model's Rule recognition step matches the dataset's flipped rule.

Score only CD-Score. Do not judge the final answer or the reasoning step.

Rubric:
- cd_score = 1.0 if the Rule recognition step accurately captures the key mechanism, direction, condition, threshold, or entity change in the flipped rule.
- cd_score = 0.5 if it notices that the rule differs from ordinary knowledge or partially captures the flipped rule, but misses important details.
- cd_score = 0.0 if it misses the flipped rule, restates the ordinary rule, reverses the flipped rule, or gives only unrelated visual/reasoning content.

Return JSON only:
{{
  "cd_score": 0.5
}}

Original ordinary rule: {original_rule}

Dataset flipped_rule: {sample.flipped_rule}

Rule to judge:
{rule_recognition_text}
"""


def _load_samples_for_predictions(run_config: Dict[str, Any], predictions: List[Dict[str, Any]]) -> Dict[str, Stage3ConflictSample]:
    sample_ids = [str(row.get("sample_id", "")).strip() for row in predictions if str(row.get("sample_id", "")).strip()]
    samples, _ = load_stage3_conflict_samples(
        data_file=run_config["data_file"],
        image_root=run_config["image_root"],
        captions_file=run_config.get("captions_file"),
        split=None if run_config.get("split") == "all" else run_config.get("split"),
        sample_ids=sample_ids,
        task_variant=run_config.get("task_variant", "conflict"),
        distractor_rule_intensities=run_config.get("distractor_rule_intensity"),
    )
    return {sample.sample_id: sample for sample in samples}


def _build_skip_record(prediction: Dict[str, Any], sample: Optional[Stage3ConflictSample], judge_status: str, error: str) -> Dict[str, Any]:
    return {
        "sample_id": prediction.get("sample_id"),
        "qid": prediction.get("qid") or (sample.qid if sample else ""),
        "conflict_intensity": prediction.get("conflict_intensity") or (sample.conflict_intensity if sample else ""),
        "causal_type": prediction.get("causal_type") or (sample.causal_type if sample else ""),
        "is_correct": bool(prediction.get("is_correct")),
        "judge_status": judge_status,
        "error": error,
    }


def _judge_prediction(
    client: InferenceAPIClient,
    sample: Stage3ConflictSample,
    prediction: Dict[str, Any],
    prompt_mode: str,
) -> Dict[str, Any]:
    if prediction.get("status") != "ok":
        return _build_skip_record(prediction, sample, "source_not_ok", "Prediction status is not ok.")

    prediction = _with_reparsed_ca_cot_steps(prediction, prompt_mode)
    if not prediction.get("ca_cot_has_complete_steps", True):
        return _build_skip_record(prediction, sample, "incomplete_ca_cot", "CA-CoT steps are incomplete.")

    rule_step_name, rule_recognition_text = _select_rule_recognition_step(prompt_mode, prediction)
    if not rule_recognition_text:
        return _build_skip_record(prediction, sample, "missing_rule_recognition", f"{rule_step_name} is empty.")

    prompt_text = _build_judge_prompt(sample, prediction, prompt_mode)
    content_blocks = [
        {"type": "text", "text": prompt_text},
    ]
    raw_output = client.generate(system_message=SYSTEM_MESSAGE, content_blocks=content_blocks)
    parsed = _parse_judge_output(raw_output)
    matches_factual_baseline = prediction.get("matches_factual_baseline")
    quadrant = derive_quadrant(parsed["cd_score"], bool(prediction.get("is_correct")))
    error_type = derive_error_type(
        bool(prediction.get("is_correct")),
        parsed["cd_score"],
        matches_factual_baseline if isinstance(matches_factual_baseline, bool) else None,
    )
    return {
        "sample_id": prediction.get("sample_id"),
        "qid": sample.qid,
        "conflict_intensity": sample.conflict_intensity,
        "causal_type": sample.causal_type,
        "is_correct": bool(prediction.get("is_correct")),
        "judge_status": "ok",
        "cd_score": parsed["cd_score"],
        "flipped_rule": sample.flipped_rule,
        "rule_recognition_step": rule_step_name,
        "rule_recognition_text": rule_recognition_text,
        "rule_exact_match": parsed["cd_score"] == 1.0,
        "rule_conflict_detected": parsed["cd_score"] >= 0.5,
        "matches_factual_baseline": matches_factual_baseline,
        "quadrant": quadrant,
        "error_type": error_type,
        "raw_judge_output": raw_output,
    }


def _load_cached_records(path: Path) -> Dict[str, Dict[str, Any]]:
    cached: Dict[str, Dict[str, Any]] = {}
    for record in load_jsonl(path):
        sample_id = str(record.get("sample_id", "")).strip()
        if sample_id:
            cached[sample_id] = record
    return cached


def _append_record(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def _print_dry_run_preview(
    client: InferenceAPIClient,
    predictions: List[Dict[str, Any]],
    samples_by_id: Dict[str, Stage3ConflictSample],
    prompt_mode: str,
) -> int:
    print("# CA-CoT Judge Dry Run")
    print(f"prompt_mode: {prompt_mode}")
    print(f"records: {len(predictions)}")
    for index, prediction in enumerate(predictions, start=1):
        prediction = _with_reparsed_ca_cot_steps(prediction, prompt_mode)
        sample_id = str(prediction.get("sample_id", "")).strip()
        print("")
        print(f"## Record {index}")
        print(f"sample_id: {sample_id}")
        sample = samples_by_id.get(sample_id)
        if sample is None:
            print("status: missing_sample")
            print("error: Sample could not be reloaded.")
            continue
        rule_step_name, rule_recognition_text = _select_rule_recognition_step(prompt_mode, prediction)
        print(f"qid: {sample.qid}")
        print(f"rule_recognition_step: {rule_step_name}")
        print("Rule recognition text:")
        print(rule_recognition_text)
        print("Dataset flipped_rule:")
        print(sample.flipped_rule)
        prompt = _build_judge_prompt(sample, prediction, prompt_mode)
        print("Judge prompt:")
        print(prompt)
        content_blocks = [{"type": "text", "text": prompt}]
        payload = client.build_payload(system_message=SYSTEM_MESSAGE, content_blocks=content_blocks)
        print("Dry Run Endpoint:")
        print(client._get_endpoint_url())
        print("Dry Run Headers (redacted):")
        print(json.dumps(_redact_headers_for_preview(client._build_headers()), indent=2, ensure_ascii=False))
        print("Dry Run Payload (redacted):")
        print(json.dumps(_redact_payload_for_preview(payload), indent=2, ensure_ascii=False))
    return len(predictions)


def run_judge(args: argparse.Namespace) -> Dict[str, Any]:
    context = load_run_context(args.run_dir)
    run_config = context["run_config"]
    prompt_mode = str(run_config.get("prompt_mode", ""))
    if not is_ca_cot_prompt_mode(prompt_mode):
        raise ValueError("Judge script only supports runs produced with CA-CoT-style prompt modes.")

    run_dir = Path(args.run_dir)
    output_dir = resolve_judge_output_dir(run_dir, args.judge_model)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "judge_records.jsonl"
    cached = _load_cached_records(records_path) if args.resume else {}
    if not args.resume and records_path.exists():
        records_path.unlink()

    predictions = context["predictions"]
    if args.limit is not None:
        predictions = predictions[: args.limit]

    samples_by_id = _load_samples_for_predictions(run_config, predictions)
    if getattr(args, "dry_run", False):
        dry_run_records = _print_dry_run_preview(_build_preview_client(args), predictions, samples_by_id, prompt_mode)
        return {
            "output_dir": str(output_dir),
            "dry_run": True,
            "dry_run_records": dry_run_records,
        }

    client = _build_client(args)

    def judge_one(prediction: Dict[str, Any]) -> Dict[str, Any]:
        sample_id = str(prediction.get("sample_id", "")).strip()
        if args.resume and cached.get(sample_id, {}).get("judge_status") == "ok":
            return cached[sample_id]
        sample = samples_by_id.get(sample_id)
        if sample is None:
            return _build_skip_record(prediction, None, "missing_sample", "Sample could not be reloaded.")
        try:
            return _judge_prediction(client, sample, prediction, prompt_mode)
        except (APIClientError, ValueError) as exc:
            return _build_skip_record(prediction, sample, "judge_error", str(exc))

    workers = max(1, int(getattr(args, "workers", 1) or 1))
    results_by_sample_id: Dict[str, Dict[str, Any]] = {}
    if workers == 1:
        for prediction in predictions:
            record = judge_one(prediction)
            sample_id = str(record.get("sample_id", "")).strip()
            if not (args.resume and cached.get(sample_id, {}).get("judge_status") == "ok"):
                _append_record(records_path, record)
            if sample_id:
                results_by_sample_id[sample_id] = record
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_prediction = {executor.submit(judge_one, prediction): prediction for prediction in predictions}
            for future in as_completed(future_to_prediction):
                record = future.result()
                sample_id = str(record.get("sample_id", "")).strip()
                if not (args.resume and cached.get(sample_id, {}).get("judge_status") == "ok"):
                    _append_record(records_path, record)
                if sample_id:
                    results_by_sample_id[sample_id] = record

    results = []
    for prediction in predictions:
        sample_id = str(prediction.get("sample_id", "")).strip()
        record = results_by_sample_id.get(sample_id)
        if record is not None:
            results.append(record)

    bundle = write_ca_cot_judge_report_bundle(
        output_dir=output_dir,
        records=results,
        judge_config={
            "judge_model": args.judge_model,
            "judge_api_base_url": args.judge_api_base_url,
            "judge_endpoint_type": args.judge_endpoint_type,
            "judge_reasoning_effort": args.judge_reasoning_effort,
        },
        run_config=run_config,
    )
    return {"output_dir": str(output_dir), "summary": bundle["summary"]}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = run_judge(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
