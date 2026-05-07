from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


OPTION_LABELS = [chr(code) for code in range(ord("A"), ord("Z") + 1)]


@dataclass(frozen=True)
class Stage3ConflictSample:
    sample_id: str
    qid: str
    flip_index: int
    task_variant: str
    split: str
    topic: str
    category: str
    subject: str
    causal_type: str
    conflict_intensity: str
    question: str
    choices: List[str]
    answer_idx: int
    answer_text: str
    option_labels: List[str]
    factual_baseline_index: Optional[int]
    factual_baseline_text: str
    flipped_rule: str
    flipped_causal_chain: str
    image_name: str
    image_path: Path
    image_caption: str
    source_record: Dict[str, Any]
    hop_count: Optional[int] = None
    reasoning_complexity: Optional[str] = None
    donor_qid: Optional[str] = None
    donor_topic: Optional[str] = None
    donor_category: Optional[str] = None


def load_json(path: Path | str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as file_obj:
        return json.load(file_obj)


def load_captions(path: Path | str | None) -> Dict[str, str]:
    if path is None:
        return {}

    payload = load_json(path)
    raw_captions = payload.get("captions", payload)
    if not isinstance(raw_captions, dict):
        return {}

    captions: Dict[str, str] = {}
    for key, value in raw_captions.items():
        if value is None:
            continue
        text = str(value).strip()
        if text:
            captions[str(key)] = text
    return captions


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
            if line:
                records.append(json.loads(line))
    return records


def _normalize_filter_values(values: Optional[Sequence[str]]) -> Optional[set[str]]:
    if not values:
        return None
    return {value.strip() for value in values if value and value.strip()}


def _matches_filter(value: str, allowed_values: Optional[set[str]]) -> bool:
    if allowed_values is None:
        return True
    return value in allowed_values


def resolve_problem_image_path(problem: Dict[str, Any], qid: str, image_root: Path | str) -> Path:
    return Path(image_root) / str(problem["split"]) / str(qid) / str(problem["image"])


def _validate_problem(problem: Dict[str, Any], qid: str, image_root: Path | str) -> List[str]:
    errors: List[str] = []
    for field in ["split", "image", "topic", "category", "subject", "causal_type", "flips"]:
        if field not in problem:
            errors.append(f"missing_field:{field}")

    flips = problem.get("flips")
    if not isinstance(flips, list) or not flips:
        errors.append("invalid_flips")

    image_name = problem.get("image")
    split = problem.get("split")
    if image_name and split:
        image_path = resolve_problem_image_path(problem, qid, image_root)
        if not image_path.exists():
            errors.append(f"missing_image:{image_path}")

    return errors


def _validate_causal_fields(problem: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for field in ["causal_question", "causal_choices", "causal_answer"]:
        if field not in problem:
            errors.append(f"missing_field:{field}")

    choices = problem.get("causal_choices")
    if not isinstance(choices, list) or not choices:
        errors.append("invalid_causal_choices")
    elif len(choices) > len(OPTION_LABELS):
        errors.append("too_many_causal_choices")

    answer_idx = problem.get("causal_answer")
    if not isinstance(answer_idx, int):
        errors.append("invalid_causal_answer_type")
    elif isinstance(choices, list) and (answer_idx < 0 or answer_idx >= len(choices)):
        errors.append("causal_answer_out_of_bounds")

    question = str(problem.get("causal_question", "")).strip()
    if not question:
        errors.append("empty_causal_question")

    return errors


def _validate_flip(flip: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for field in [
        "conflict_intensity",
        "flipped_rule",
        "conflict_question",
        "conflict_choices",
        "conflict_answer",
        "factual_baseline_index",
        "flipped_causal_chain",
    ]:
        if field not in flip:
            errors.append(f"missing_flip_field:{field}")

    choices = flip.get("conflict_choices")
    if not isinstance(choices, list) or not choices:
        errors.append("invalid_conflict_choices")
    elif len(choices) > len(OPTION_LABELS):
        errors.append("too_many_conflict_choices")

    answer_idx = flip.get("conflict_answer")
    if not isinstance(answer_idx, int):
        errors.append("invalid_conflict_answer_type")
    elif isinstance(choices, list) and (answer_idx < 0 or answer_idx >= len(choices)):
        errors.append("conflict_answer_out_of_bounds")

    baseline_idx = flip.get("factual_baseline_index")
    if not isinstance(baseline_idx, int):
        errors.append("invalid_factual_baseline_index_type")
    elif isinstance(choices, list) and (baseline_idx < 0 or baseline_idx >= len(choices)):
        errors.append("factual_baseline_index_out_of_bounds")

    return errors


def map_hop_count_to_reasoning_complexity(hop_count: Optional[int]) -> Optional[str]:
    if hop_count is None:
        return None
    if hop_count <= 1:
        return "R1"
    if hop_count == 2:
        return "R2"
    return "R3"


def build_conflict_sample(
    problem: Dict[str, Any],
    qid: str,
    flip: Dict[str, Any],
    flip_index: int,
    image_root: Path | str,
    image_caption: str = "",
) -> Stage3ConflictSample:
    choices = [str(choice).strip() for choice in flip["conflict_choices"]]
    answer_idx = int(flip["conflict_answer"])
    factual_baseline_index = int(flip["factual_baseline_index"])
    hop_count_raw = flip.get("hop_count")
    hop_count = int(hop_count_raw) if isinstance(hop_count_raw, int) else None
    option_labels = OPTION_LABELS[: len(choices)]
    conflict_intensity = str(flip["conflict_intensity"]).strip()
    sample_id = f"{qid}__{conflict_intensity}__{flip_index}"
    return Stage3ConflictSample(
        sample_id=sample_id,
        qid=str(qid),
        flip_index=flip_index,
        task_variant="conflict",
        split=str(problem["split"]).strip(),
        topic=str(problem.get("topic", "")).strip(),
        category=str(problem.get("category", "")).strip(),
        subject=str(problem.get("subject", "")).strip(),
        causal_type=str(problem.get("causal_type", "")).strip().lower(),
        conflict_intensity=conflict_intensity,
        question=str(flip["conflict_question"]).strip(),
        choices=choices,
        answer_idx=answer_idx,
        answer_text=choices[answer_idx],
        option_labels=option_labels,
        factual_baseline_index=factual_baseline_index,
        factual_baseline_text=choices[factual_baseline_index],
        flipped_rule=str(flip["flipped_rule"]).strip(),
        flipped_causal_chain=str(flip["flipped_causal_chain"]).strip(),
        hop_count=hop_count,
        reasoning_complexity=map_hop_count_to_reasoning_complexity(hop_count),
        image_name=str(problem["image"]).strip(),
        image_path=resolve_problem_image_path(problem, qid, image_root),
        image_caption=image_caption,
        source_record={"problem": dict(problem), "flip": dict(flip)},
    )


def _build_donor_index(dataset: Dict[str, Any], image_root: Path | str) -> Dict[str, List[Dict[str, Any]]]:
    donor_index: Dict[str, List[Dict[str, Any]]] = {}
    for donor_qid, donor_problem in dataset.items():
        problem_errors = _validate_problem(donor_problem, str(donor_qid), image_root)
        if problem_errors:
            continue

        donor_topic = str(donor_problem.get("topic", "")).strip()
        donor_category = str(donor_problem.get("category", "")).strip()
        for donor_flip_index, donor_flip in enumerate(donor_problem.get("flips", [])):
            if _validate_flip(donor_flip):
                continue

            candidate = {
                "donor_qid": str(donor_qid),
                "donor_flip_index": donor_flip_index,
                "donor_problem": donor_problem,
                "donor_flip": donor_flip,
                "donor_topic": donor_topic,
                "donor_category": donor_category,
            }
            donor_index.setdefault(str(donor_flip.get("conflict_intensity", "")).strip(), []).append(candidate)

    for entries in donor_index.values():
        entries.sort(key=lambda item: (item["donor_qid"], item["donor_flip_index"]))
    return donor_index


def _select_donor_candidate(
    donor_index: Dict[str, List[Dict[str, Any]]],
    intensity: str,
    target_qid: str,
    target_topic: str,
    target_category: str,
) -> Optional[Dict[str, Any]]:
    candidates = donor_index.get(intensity, [])
    if not candidates:
        return None

    rng = random.Random(f"{target_qid}:{intensity}")
    start_offset = rng.randrange(len(candidates))

    for require_topic_diff in (True, False):
        for offset in range(len(candidates)):
            candidate = candidates[(start_offset + offset) % len(candidates)]
            if candidate["donor_qid"] == target_qid:
                continue
            if candidate["donor_category"] == target_category:
                continue
            if require_topic_diff and candidate["donor_topic"] == target_topic:
                continue
            return candidate
    return None


def build_distractor_sample(
    problem: Dict[str, Any],
    qid: str,
    donor_problem: Dict[str, Any],
    donor_flip: Dict[str, Any],
    donor_flip_index: int,
    image_root: Path | str,
    image_caption: str = "",
) -> Stage3ConflictSample:
    choices = [str(choice).strip() for choice in problem["causal_choices"]]
    answer_idx = int(problem["causal_answer"])
    option_labels = OPTION_LABELS[: len(choices)]
    intensity = str(donor_flip["conflict_intensity"]).strip()
    hop_count_raw = donor_flip.get("hop_count")
    hop_count = int(hop_count_raw) if isinstance(hop_count_raw, int) else None
    donor_qid = str(donor_problem.get("_qid", "")).strip()
    sample_id = f"{qid}__T4__{intensity}__{donor_qid}_{donor_flip_index}"
    return Stage3ConflictSample(
        sample_id=sample_id,
        qid=str(qid),
        flip_index=donor_flip_index,
        task_variant="distractor",
        split=str(problem["split"]).strip(),
        topic=str(problem.get("topic", "")).strip(),
        category=str(problem.get("category", "")).strip(),
        subject=str(problem.get("subject", "")).strip(),
        causal_type=str(problem.get("causal_type", "")).strip().lower(),
        conflict_intensity=intensity,
        question=str(problem["causal_question"]).strip(),
        choices=choices,
        answer_idx=answer_idx,
        answer_text=choices[answer_idx],
        option_labels=option_labels,
        factual_baseline_index=answer_idx,
        factual_baseline_text=choices[answer_idx],
        flipped_rule=str(donor_flip["flipped_rule"]).strip(),
        flipped_causal_chain="",
        hop_count=hop_count,
        reasoning_complexity=map_hop_count_to_reasoning_complexity(hop_count),
        image_name=str(problem["image"]).strip(),
        image_path=resolve_problem_image_path(problem, qid, image_root),
        image_caption=image_caption,
        source_record={"problem": dict(problem), "donor_problem": dict(donor_problem), "donor_flip": dict(donor_flip)},
        donor_qid=donor_qid,
        donor_topic=str(donor_problem.get("topic", "")).strip(),
        donor_category=str(donor_problem.get("category", "")).strip(),
    )


def load_stage3_conflict_samples(
    data_file: Path | str,
    image_root: Path | str,
    captions_file: Path | str | None = None,
    split: Optional[str] = None,
    limit: Optional[int] = None,
    topics: Optional[Sequence[str]] = None,
    categories: Optional[Sequence[str]] = None,
    causal_types: Optional[Sequence[str]] = None,
    conflict_intensities: Optional[Sequence[str]] = None,
    qids: Optional[Sequence[str]] = None,
    sample_ids: Optional[Sequence[str]] = None,
    task_variant: str = "conflict",
    distractor_rule_intensities: Optional[Sequence[str]] = None,
) -> tuple[List[Stage3ConflictSample], Dict[str, Any]]:
    if task_variant not in {"conflict", "distractor"}:
        raise ValueError(f"Unsupported task_variant: {task_variant}")

    dataset = load_json(data_file)
    captions = load_captions(captions_file)
    donor_index = _build_donor_index(dataset, image_root) if task_variant == "distractor" else {}
    allowed_topics = _normalize_filter_values(topics)
    allowed_categories = _normalize_filter_values(categories)
    allowed_causal_types = _normalize_filter_values(causal_types)
    allowed_conflict_intensities = _normalize_filter_values(conflict_intensities)
    allowed_qids = _normalize_filter_values(qids)
    allowed_sample_ids = _normalize_filter_values(sample_ids)
    allowed_distractor_intensities = _normalize_filter_values(distractor_rule_intensities)

    samples: List[Stage3ConflictSample] = []
    invalid_records: List[Dict[str, Any]] = []

    for qid, problem in dataset.items():
        if isinstance(problem, dict):
            problem["_qid"] = str(qid)

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

        problem_errors = _validate_problem(problem, qid, image_root)
        if problem_errors:
            invalid_records.append({"qid": str(qid), "errors": problem_errors})
            continue

        if task_variant == "conflict":
            for flip_index, flip in enumerate(problem["flips"]):
                flip_errors = _validate_flip(flip)
                if flip_errors:
                    invalid_records.append({"qid": str(qid), "flip_index": flip_index, "errors": flip_errors})
                    continue

                sample = build_conflict_sample(
                    problem,
                    qid,
                    flip,
                    flip_index,
                    image_root,
                    image_caption=captions.get(str(qid), ""),
                )
                if not _matches_filter(sample.conflict_intensity, allowed_conflict_intensities):
                    continue
                if allowed_sample_ids is not None and sample.sample_id not in allowed_sample_ids:
                    continue

                samples.append(sample)
                if limit is not None and limit > 0 and len(samples) >= limit:
                    return samples, build_preflight_report(samples, invalid_records)
            continue

        causal_errors = _validate_causal_fields(problem)
        if causal_errors:
            invalid_records.append({"qid": str(qid), "errors": causal_errors})
            continue

        requested_intensities = sorted(
            allowed_distractor_intensities or {"C1", "C2", "C3"},
            key=lambda value: ("C1", "C2", "C3").index(value) if value in {"C1", "C2", "C3"} else 99,
        )
        for intensity in requested_intensities:
            donor = _select_donor_candidate(
                donor_index=donor_index,
                intensity=intensity,
                target_qid=str(qid),
                target_topic=str(problem.get("topic", "")).strip(),
                target_category=str(problem.get("category", "")).strip(),
            )
            if donor is None:
                invalid_records.append(
                    {
                        "qid": str(qid),
                        "task_variant": "distractor",
                        "conflict_intensity": intensity,
                        "errors": ["missing_donor_candidate"],
                    }
                )
                continue

            sample = build_distractor_sample(
                problem=problem,
                qid=str(qid),
                donor_problem=donor["donor_problem"],
                donor_flip=donor["donor_flip"],
                donor_flip_index=int(donor["donor_flip_index"]),
                image_root=image_root,
                image_caption=captions.get(str(qid), ""),
            )
            if allowed_sample_ids is not None and sample.sample_id not in allowed_sample_ids:
                continue

            samples.append(sample)
            if limit is not None and limit > 0 and len(samples) >= limit:
                return samples, build_preflight_report(samples, invalid_records)

    return samples, build_preflight_report(samples, invalid_records)


def build_preflight_report(samples: Sequence[Stage3ConflictSample], invalid_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    split_counts: Dict[str, int] = {}
    topic_counts: Dict[str, int] = {}
    category_counts: Dict[str, int] = {}
    causal_type_counts: Dict[str, int] = {}
    conflict_intensity_counts: Dict[str, int] = {}
    reasoning_complexity_counts: Dict[str, int] = {}

    for sample in samples:
        split_counts[sample.split] = split_counts.get(sample.split, 0) + 1
        topic_counts[sample.topic] = topic_counts.get(sample.topic, 0) + 1
        category_counts[sample.category] = category_counts.get(sample.category, 0) + 1
        causal_type_counts[sample.causal_type] = causal_type_counts.get(sample.causal_type, 0) + 1
        conflict_intensity_counts[sample.conflict_intensity] = conflict_intensity_counts.get(sample.conflict_intensity, 0) + 1
        if sample.reasoning_complexity:
            reasoning_complexity_counts[sample.reasoning_complexity] = reasoning_complexity_counts.get(sample.reasoning_complexity, 0) + 1

    return {
        "valid_sample_count": len(samples),
        "invalid_sample_count": len(invalid_records),
        "split_counts": split_counts,
        "topic_counts": topic_counts,
        "category_counts": category_counts,
        "causal_type_counts": causal_type_counts,
        "conflict_intensity_counts": conflict_intensity_counts,
        "reasoning_complexity_counts": reasoning_complexity_counts,
        "invalid_records_preview": list(invalid_records[:50]),
    }
