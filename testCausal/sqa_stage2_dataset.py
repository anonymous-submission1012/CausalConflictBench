from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


OPTION_LABELS = [chr(code) for code in range(ord("A"), ord("Z") + 1)]


@dataclass(frozen=True)
class Stage2Sample:
    qid: str
    split: str
    question: str
    choices: List[str]
    answer_idx: int
    answer_text: str
    option_labels: List[str]
    hint: str
    topic: str
    category: str
    subject: str
    causal_type: str
    causal_chain: str
    requires_image: bool
    image_dependency_reason: str
    image_name: str
    image_path: Path
    has_choice_images: bool
    choice_image_paths: List[Path]
    stage2_quality_action: str
    source_record: Dict[str, Any]


def load_json(path: Path | str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as file_obj:
        return json.load(file_obj)


def save_json(path: Path | str, payload: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)


def append_jsonl(path: Path | str, records: Iterable[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return []

    records: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8-sig") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def resolve_image_path(problem: Dict[str, Any], qid: str, image_root: Path | str) -> Path:
    return Path(image_root) / str(problem["split"]) / str(qid) / str(problem["image"])


def resolve_choice_image_paths(problem: Dict[str, Any], qid: str, image_root: Path | str) -> List[Path]:
    sample_dir = Path(image_root) / str(problem["split"]) / str(qid)
    if not sample_dir.exists():
        return []
    return sorted(sample_dir.glob("choice_*.png"))


def _normalize_filter_values(values: Optional[Sequence[str]]) -> Optional[set[str]]:
    if not values:
        return None
    return {value.strip() for value in values if value and value.strip()}


def _matches_filter(value: str, allowed_values: Optional[set[str]]) -> bool:
    if allowed_values is None:
        return True
    return value in allowed_values


def validate_problem_record(problem: Dict[str, Any], qid: str, image_root: Path | str) -> List[str]:
    errors: List[str] = []
    required_fields = [
        "split",
        "image",
        "causal_question",
        "causal_choices",
        "causal_answer",
        "causal_chain",
        "causal_type",
        "requires_image",
    ]
    for field in required_fields:
        if field not in problem:
            errors.append(f"missing_field:{field}")

    choices = problem.get("causal_choices")
    if not isinstance(choices, list) or not choices:
        errors.append("invalid_causal_choices")
    elif len(choices) > len(OPTION_LABELS):
        errors.append("too_many_choices")

    answer_idx = problem.get("causal_answer")
    if not isinstance(answer_idx, int):
        errors.append("invalid_causal_answer_type")
    elif isinstance(choices, list) and (answer_idx < 0 or answer_idx >= len(choices)):
        errors.append("causal_answer_out_of_bounds")

    causal_type = str(problem.get("causal_type", "")).strip().lower()
    if causal_type not in {"effect", "cause"}:
        errors.append("invalid_causal_type")

    image_name = problem.get("image")
    split = problem.get("split")
    if image_name and split:
        image_path = resolve_image_path(problem, qid, image_root)
        if not image_path.exists():
            errors.append(f"missing_image:{image_path}")

    if problem.get("requires_image") is not True:
        errors.append("requires_image_not_true")

    question = str(problem.get("causal_question", "")).strip()
    if not question:
        errors.append("empty_causal_question")

    chain = str(problem.get("causal_chain", "")).strip()
    if not chain:
        errors.append("empty_causal_chain")

    return errors


def build_sample(problem: Dict[str, Any], qid: str, image_root: Path | str) -> Stage2Sample:
    choices = [str(choice).strip() for choice in problem["causal_choices"]]
    answer_idx = int(problem["causal_answer"])
    option_labels = OPTION_LABELS[: len(choices)]
    choice_image_paths = resolve_choice_image_paths(problem, qid, image_root)
    image_path = resolve_image_path(problem, qid, image_root)
    return Stage2Sample(
        qid=str(qid),
        split=str(problem["split"]),
        question=str(problem["causal_question"]).strip(),
        choices=choices,
        answer_idx=answer_idx,
        answer_text=choices[answer_idx],
        option_labels=option_labels,
        hint=str(problem.get("hint", "")).strip(),
        topic=str(problem.get("topic", "")).strip(),
        category=str(problem.get("category", "")).strip(),
        subject=str(problem.get("subject", "")).strip(),
        causal_type=str(problem.get("causal_type", "")).strip().lower(),
        causal_chain=str(problem.get("causal_chain", "")).strip(),
        requires_image=bool(problem.get("requires_image", False)),
        image_dependency_reason=str(problem.get("image_dependency_reason", "")).strip(),
        image_name=str(problem["image"]),
        image_path=image_path,
        has_choice_images=bool(choice_image_paths),
        choice_image_paths=choice_image_paths,
        stage2_quality_action=str(problem.get("stage2_quality_action", "")).strip(),
        source_record=dict(problem),
    )


def load_stage2_samples(
    data_file: Path | str,
    image_root: Path | str,
    split: Optional[str] = None,
    limit: Optional[int] = None,
    topics: Optional[Sequence[str]] = None,
    categories: Optional[Sequence[str]] = None,
    causal_types: Optional[Sequence[str]] = None,
    qids: Optional[Sequence[str]] = None,
) -> tuple[List[Stage2Sample], Dict[str, Any]]:
    dataset = load_json(data_file)
    allowed_topics = _normalize_filter_values(topics)
    allowed_categories = _normalize_filter_values(categories)
    allowed_causal_types = _normalize_filter_values(causal_types)
    allowed_qids = _normalize_filter_values(qids)

    samples: List[Stage2Sample] = []
    invalid_records: List[Dict[str, Any]] = []

    for qid, problem in dataset.items():
        if split and str(problem.get("split")) != split:
            continue
        if allowed_qids is not None and str(qid) not in allowed_qids:
            continue
        if not _matches_filter(str(problem.get("topic", "")).strip(), allowed_topics):
            continue
        if not _matches_filter(str(problem.get("category", "")).strip(), allowed_categories):
            continue
        if not _matches_filter(str(problem.get("causal_type", "")).strip().lower(), allowed_causal_types):
            continue

        errors = validate_problem_record(problem, qid, image_root)
        if errors:
            invalid_records.append(
                {
                    "qid": str(qid),
                    "errors": errors,
                    "split": str(problem.get("split", "")),
                    "topic": str(problem.get("topic", "")),
                    "category": str(problem.get("category", "")),
                }
            )
            continue

        samples.append(build_sample(problem, qid, image_root))
        if limit is not None and limit > 0 and len(samples) >= limit:
            break

    preflight_report = build_preflight_report(samples, invalid_records)
    return samples, preflight_report


def build_preflight_report(samples: Sequence[Stage2Sample], invalid_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    split_counts: Dict[str, int] = {}
    topic_counts: Dict[str, int] = {}
    causal_type_counts: Dict[str, int] = {}
    choice_count_counts: Dict[str, int] = {}
    missing_choice_image_qids: List[str] = []

    for sample in samples:
        split_counts[sample.split] = split_counts.get(sample.split, 0) + 1
        topic_counts[sample.topic] = topic_counts.get(sample.topic, 0) + 1
        causal_type_counts[sample.causal_type] = causal_type_counts.get(sample.causal_type, 0) + 1
        choice_key = str(len(sample.choices))
        choice_count_counts[choice_key] = choice_count_counts.get(choice_key, 0) + 1
        if sample.has_choice_images:
            missing_choice_image_qids.append(sample.qid)

    return {
        "valid_sample_count": len(samples),
        "invalid_sample_count": len(invalid_records),
        "split_counts": split_counts,
        "topic_counts": topic_counts,
        "causal_type_counts": causal_type_counts,
        "choice_count_counts": choice_count_counts,
        "samples_with_choice_images": len(missing_choice_image_qids),
        "sample_qids_with_choice_images_preview": missing_choice_image_qids[:50],
        "invalid_records_preview": list(invalid_records[:50]),
    }
