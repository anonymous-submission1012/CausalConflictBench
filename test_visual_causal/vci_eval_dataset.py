from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


OPTION_LABELS = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
_SAMPLE_KEY_SEP = "::"


def build_sample_key(domain: str, qid: str) -> str:
    return f"{str(domain).strip()}{_SAMPLE_KEY_SEP}{str(qid).strip()}"


@dataclass(frozen=True)
class VciSample:
    qid: str
    domain: str
    question_scope: str
    mechanism_id: str
    mechanism_name: str
    variant_id: str
    transfer_type: str
    question: str
    choices: List[str]
    answer_idx: int
    answer_text: str
    option_labels: List[str]
    answer_label: str
    answer_prior: str
    visual_context: str
    required_induction: str
    reasoning_chain: str
    image_path: Path
    image_name: str
    source_record: Dict[str, Any]
    frame_descriptions: Dict[str, str] = field(default_factory=dict)

    @property
    def sample_key(self) -> str:
        return build_sample_key(self.domain, self.qid)


@dataclass(frozen=True)
class VciDomainExclusionSpec:
    qids: frozenset[str] = frozenset()
    mechanism_ids: frozenset[str] = frozenset()
    variant_ids: frozenset[str] = frozenset()
    exclude_all: bool = False


@dataclass(frozen=True)
class VciSampleExclusionSpec:
    qids: frozenset[str] = frozenset()
    mechanism_ids: frozenset[str] = frozenset()
    variant_ids: frozenset[str] = frozenset()
    domains: Dict[str, VciDomainExclusionSpec] = field(default_factory=dict)

    def collect_reasons(self, domain: str, record: Dict[str, Any]) -> List[str]:
        reasons: List[str] = []
        qid = str(record.get("question_id", "")).strip()
        mechanism_id = str(record.get("mechanism_id", "")).strip()
        variant_id = str(record.get("variant_id", "")).strip()

        if qid and qid in self.qids:
            reasons.append(f"qid:{qid}")
        if mechanism_id and mechanism_id in self.mechanism_ids:
            reasons.append(f"mechanism_id:{mechanism_id}")
        if variant_id and variant_id in self.variant_ids:
            reasons.append(f"variant_id:{variant_id}")

        domain_spec = self.domains.get(domain)
        if not domain_spec:
            return reasons

        if domain_spec.exclude_all:
            reasons.append(f"domain:{domain}")
        if qid and qid in domain_spec.qids:
            reasons.append(f"{domain}::qid:{qid}")
        if mechanism_id and mechanism_id in domain_spec.mechanism_ids:
            reasons.append(f"{domain}::mechanism_id:{mechanism_id}")
        if variant_id and variant_id in domain_spec.variant_ids:
            reasons.append(f"{domain}::variant_id:{variant_id}")
        return reasons


def load_json(path: Path | str) -> Any:
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


def _normalize_filter_values(values: Optional[Sequence[str]]) -> Optional[set[str]]:
    if not values:
        return None
    return {value.strip() for value in values if value and value.strip()}


def _normalize_string_values(values: Any, field_name: str) -> frozenset[str]:
    if values is None:
        return frozenset()
    if isinstance(values, str):
        raw_values: Sequence[Any] = [values]
    elif isinstance(values, Sequence):
        raw_values = values
    else:
        raise ValueError(f"{field_name} must be a string or a list of strings.")

    normalized = {str(value).strip() for value in raw_values if str(value).strip()}
    return frozenset(normalized)


def _parse_domain_exclusion_spec(domain: str, payload: Any) -> VciDomainExclusionSpec:
    if isinstance(payload, bool):
        return VciDomainExclusionSpec(exclude_all=payload)
    if isinstance(payload, str) or isinstance(payload, Sequence) and not isinstance(payload, dict):
        return VciDomainExclusionSpec(
            mechanism_ids=_normalize_string_values(payload, f"domains.{domain}.mechanism_ids"),
        )
    if payload is None:
        return VciDomainExclusionSpec()
    if not isinstance(payload, dict):
        raise ValueError(f"domains.{domain} must be a string, list, bool, or object.")

    return VciDomainExclusionSpec(
        qids=_normalize_string_values(payload.get("qids"), f"domains.{domain}.qids"),
        mechanism_ids=_normalize_string_values(
            payload.get("mechanism_ids", payload.get("mechanisms")),
            f"domains.{domain}.mechanism_ids",
        ),
        variant_ids=_normalize_string_values(
            payload.get("variant_ids", payload.get("variants")),
            f"domains.{domain}.variant_ids",
        ),
        exclude_all=bool(payload.get("all")),
    )


def parse_sample_exclusion_spec(payload: Any) -> VciSampleExclusionSpec:
    if payload is None:
        return VciSampleExclusionSpec()
    if not isinstance(payload, dict):
        raise ValueError("Exclusion JSON must be an object.")

    domain_payload = payload.get("domains")
    domain_entries: Dict[str, Any] = {}
    if domain_payload is not None:
        if not isinstance(domain_payload, dict):
            raise ValueError("domains must be an object.")
        domain_entries.update(domain_payload)

    for key, value in payload.items():
        if key in {"qids", "mechanism_ids", "mechanisms", "variant_ids", "variants", "domains"}:
            continue
        domain_entries[key] = value

    normalized_domains: Dict[str, VciDomainExclusionSpec] = {}
    for domain, value in domain_entries.items():
        normalized_domain = str(domain).strip()
        if not normalized_domain:
            raise ValueError("Domain names in exclusion JSON must not be empty.")
        normalized_domains[normalized_domain] = _parse_domain_exclusion_spec(normalized_domain, value)

    return VciSampleExclusionSpec(
        qids=_normalize_string_values(payload.get("qids"), "qids"),
        mechanism_ids=_normalize_string_values(payload.get("mechanism_ids", payload.get("mechanisms")), "mechanism_ids"),
        variant_ids=_normalize_string_values(payload.get("variant_ids", payload.get("variants")), "variant_ids"),
        domains=normalized_domains,
    )


def load_sample_exclusion_spec(path: Path | str) -> VciSampleExclusionSpec:
    return parse_sample_exclusion_spec(load_json(path))


def _matches_filter(value: str, allowed_values: Optional[set[str]]) -> bool:
    if allowed_values is None:
        return True
    return value in allowed_values


def _normalize_choices(raw_choices: Any) -> List[str]:
    if isinstance(raw_choices, dict):
        ordered_keys = [key for key in OPTION_LABELS if key in raw_choices]
        return [str(raw_choices[key]).strip() for key in ordered_keys]
    if isinstance(raw_choices, list):
        return [str(choice).strip() for choice in raw_choices]
    return []


def _normalize_frame_descriptions(raw_frame_descriptions: Any) -> Dict[str, str]:
    if isinstance(raw_frame_descriptions, dict):
        normalized: Dict[str, str] = {}
        for key, value in raw_frame_descriptions.items():
            normalized_key = str(key).strip()
            normalized_value = str(value).strip()
            if normalized_key and normalized_value:
                normalized[normalized_key] = normalized_value
        return normalized
    if isinstance(raw_frame_descriptions, list):
        return {
            f"panel_{index}": str(value).strip()
            for index, value in enumerate(raw_frame_descriptions, start=1)
            if str(value).strip()
        }
    if isinstance(raw_frame_descriptions, str) and raw_frame_descriptions.strip():
        return {"description": raw_frame_descriptions.strip()}
    return {}


def _resolve_answer_idx(answer_value: Any, choices: Sequence[str]) -> Optional[int]:
    if isinstance(answer_value, int):
        if 0 <= answer_value < len(choices):
            return answer_value
        return None
    answer_label = str(answer_value).strip().upper()
    if answer_label in OPTION_LABELS[: len(choices)]:
        return OPTION_LABELS.index(answer_label)
    return None


def resolve_variant_image_path(image_dir: Path, variant_id: str) -> Optional[Path]:
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = image_dir / f"{variant_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def validate_vci_record(record: Dict[str, Any], image_dir: Path) -> List[str]:
    errors: List[str] = []
    required_fields = [
        "question_id",
        "domain",
        "mechanism_id",
        "mechanism_name",
        "variant_id",
        "question",
        "choices",
        "answer",
        "question_scope",
        "transfer_type",
    ]
    for field in required_fields:
        if field not in record:
            errors.append(f"missing_field:{field}")

    choices = _normalize_choices(record.get("choices"))
    if not choices:
        errors.append("invalid_choices")
    elif len(choices) > len(OPTION_LABELS):
        errors.append("too_many_choices")

    answer_idx = _resolve_answer_idx(record.get("answer"), choices)
    if answer_idx is None:
        errors.append("invalid_answer")

    question_scope = str(record.get("question_scope", "")).strip().lower()
    if question_scope not in {"mechanism", "variant"}:
        errors.append("invalid_question_scope")

    variant_id = str(record.get("variant_id", "")).strip()
    if not variant_id:
        errors.append("empty_variant_id")
    else:
        image_path = resolve_variant_image_path(image_dir, variant_id)
        if image_path is None:
            errors.append(f"missing_image:{image_dir / variant_id}")

    if not str(record.get("question", "")).strip():
        errors.append("empty_question")
    return errors


def build_sample(record: Dict[str, Any], image_dir: Path) -> VciSample:
    choices = _normalize_choices(record["choices"])
    answer_idx = _resolve_answer_idx(record["answer"], choices)
    if answer_idx is None:
        raise ValueError(f"Unable to resolve answer for question_id={record.get('question_id')}")
    image_path = resolve_variant_image_path(image_dir, str(record["variant_id"]).strip())
    if image_path is None:
        raise ValueError(f"Missing image for variant_id={record.get('variant_id')}")

    option_labels = OPTION_LABELS[: len(choices)]
    visual_context = str(record.get("visual_context", "")).strip()

    return VciSample(
        qid=str(record["question_id"]).strip(),
        domain=str(record["domain"]).strip(),
        question_scope=str(record["question_scope"]).strip().lower(),
        mechanism_id=str(record["mechanism_id"]).strip(),
        mechanism_name=str(record["mechanism_name"]).strip(),
        variant_id=str(record["variant_id"]).strip(),
        transfer_type=str(record["transfer_type"]).strip(),
        question=str(record["question"]).strip(),
        choices=choices,
        answer_idx=answer_idx,
        answer_text=choices[answer_idx],
        option_labels=option_labels,
        answer_label=option_labels[answer_idx],
        answer_prior=str(record.get("answer_prior", "")).strip().upper(),
        visual_context=visual_context,
        required_induction=str(record.get("required_induction", "")).strip(),
        reasoning_chain=str(record.get("reasoning_chain", "")).strip(),
        image_path=image_path,
        image_name=image_path.name,
        source_record=dict(record),
        frame_descriptions=_normalize_frame_descriptions(record.get("frame_descriptions")),
    )


def build_preflight_report(
    samples: Sequence[VciSample],
    invalid_records: Sequence[Dict[str, Any]],
    excluded_records: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    domain_counts: Dict[str, int] = {}
    question_scope_counts: Dict[str, int] = {}
    transfer_type_counts: Dict[str, int] = {}
    excluded_reason_counts: Dict[str, int] = {}
    excluded_rows = list(excluded_records or [])

    for sample in samples:
        domain_counts[sample.domain] = domain_counts.get(sample.domain, 0) + 1
        question_scope_counts[sample.question_scope] = question_scope_counts.get(sample.question_scope, 0) + 1
        transfer_type_counts[sample.transfer_type] = transfer_type_counts.get(sample.transfer_type, 0) + 1

    for row in excluded_rows:
        for reason in row.get("excluded_by", []):
            excluded_reason_counts[str(reason)] = excluded_reason_counts.get(str(reason), 0) + 1

    return {
        "valid_sample_count": len(samples),
        "invalid_sample_count": len(invalid_records),
        "excluded_sample_count": len(excluded_rows),
        "domain_counts": domain_counts,
        "question_scope_counts": question_scope_counts,
        "transfer_type_counts": transfer_type_counts,
        "excluded_reason_counts": excluded_reason_counts,
        "invalid_records_preview": list(invalid_records[:50]),
        "excluded_records_preview": excluded_rows[:50],
    }


def load_vci_samples(
    data_root: Path | str,
    question_subdir: str = "shuffled",
    domains: Optional[Sequence[str]] = None,
    question_scopes: Optional[Sequence[str]] = None,
    transfer_types: Optional[Sequence[str]] = None,
    qids: Optional[Sequence[str]] = None,
    sample_keys: Optional[Sequence[str]] = None,
    exclusion_spec: Optional[VciSampleExclusionSpec] = None,
    limit: Optional[int] = None,
) -> tuple[List[VciSample], Dict[str, Any]]:
    data_root_path = Path(data_root)
    question_root = data_root_path / question_subdir
    image_root = data_root_path / "synthetic_image"

    allowed_domains = _normalize_filter_values(domains)
    allowed_question_scopes = _normalize_filter_values(question_scopes)
    allowed_transfer_types = _normalize_filter_values(transfer_types)
    allowed_qids = _normalize_filter_values(qids)
    allowed_sample_keys = _normalize_filter_values(sample_keys)

    samples: List[VciSample] = []
    invalid_records: List[Dict[str, Any]] = []
    excluded_records: List[Dict[str, Any]] = []

    for question_file in sorted(question_root.glob("*.json")):
        domain = question_file.stem
        if allowed_domains is not None and domain not in allowed_domains:
            continue

        image_dir = image_root / domain
        records = load_json(question_file)
        if not isinstance(records, list):
            invalid_records.append(
                {
                    "qid": f"{domain}::__file__",
                    "errors": ["invalid_file_format"],
                    "domain": domain,
                }
            )
            continue

        for record in records:
            qid = str(record.get("question_id", "")).strip()
            if allowed_qids is not None and qid not in allowed_qids:
                continue
            sample_key = build_sample_key(domain, qid)
            if allowed_sample_keys is not None and sample_key not in allowed_sample_keys:
                continue
            if not _matches_filter(str(record.get("question_scope", "")).strip().lower(), allowed_question_scopes):
                continue
            if not _matches_filter(str(record.get("transfer_type", "")).strip(), allowed_transfer_types):
                continue
            exclusion_reasons = exclusion_spec.collect_reasons(domain, record) if exclusion_spec else []
            if exclusion_reasons:
                excluded_records.append(
                    {
                        "qid": qid,
                        "domain": domain,
                        "mechanism_id": str(record.get("mechanism_id", "")).strip(),
                        "variant_id": str(record.get("variant_id", "")).strip(),
                        "question_scope": str(record.get("question_scope", "")).strip().lower(),
                        "transfer_type": str(record.get("transfer_type", "")).strip(),
                        "excluded_by": exclusion_reasons,
                    }
                )
                continue

            errors = validate_vci_record(record, image_dir)
            if errors:
                invalid_records.append(
                    {
                        "qid": qid,
                        "domain": domain,
                        "errors": errors,
                    }
                )
                continue

            samples.append(build_sample(record, image_dir))
            if limit is not None and limit > 0 and len(samples) >= limit:
                return samples, build_preflight_report(samples, invalid_records, excluded_records)

    return samples, build_preflight_report(samples, invalid_records, excluded_records)
