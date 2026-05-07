from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _safe_accuracy(correct: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(correct / total * 100.0, 4)


def _accuracy_denominator_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("status") in {"ok", "parse_error"}]


def _prior_bias_rate(rows: List[Dict[str, Any]]) -> float:
    wrong_ok_rows = [row for row in rows if row.get("status") == "ok" and not row.get("is_correct")]
    prior_hits = sum(1 for row in wrong_ok_rows if row.get("matches_answer_prior"))
    return _safe_accuracy(prior_hits, len(wrong_ok_rows))


def _variant_strict_group_key(row: Dict[str, Any]) -> Any:
    variant_id = str(row.get("variant_id", "")).strip()
    domain = str(row.get("domain", "")).strip()
    if domain:
        return domain, variant_id
    return variant_id


def _variant_strict_accuracy(rows: List[Dict[str, Any]]) -> float:
    rows_with_variant = [row for row in rows if str(row.get("variant_id", "")).strip()]
    if not rows_with_variant:
        return 0.0

    grouped_rows: Dict[Any, List[Dict[str, Any]]] = {}
    for row in rows_with_variant:
        grouped_rows.setdefault(_variant_strict_group_key(row), []).append(row)

    strict_correct = 0
    for variant_rows in grouped_rows.values():
        if all(row.get("status") == "ok" and bool(row.get("is_correct")) for row in variant_rows):
            strict_correct += 1
    return _safe_accuracy(strict_correct, len(grouped_rows))


def summarize_records(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    total = len(rows)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    accuracy_rows = _accuracy_denominator_rows(rows)
    correct = sum(1 for row in ok_rows if row.get("is_correct"))

    status_counts: Dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "total_records": total,
        "ok_records": len(ok_rows),
        "correct_records": correct,
        "accuracy": _safe_accuracy(correct, len(accuracy_rows)),
        "coverage": _safe_accuracy(len(ok_rows), total),
        "status_counts": status_counts,
        "prior_bias_rate": _prior_bias_rate(rows),
        "variant_strict_acc": _variant_strict_accuracy(rows),
        "text_match_fallback_hits": sum(1 for row in ok_rows if row.get("prediction_source") == "choice_text"),
        "letter_parse_hits": sum(1 for row in ok_rows if row.get("prediction_source") == "option_letter"),
    }


def group_metrics(records: Iterable[Dict[str, Any]], group_field: str) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in records:
        key = str(row.get(group_field, ""))
        buckets.setdefault(key, []).append(row)

    results: List[Dict[str, Any]] = []
    for key in sorted(buckets):
        rows = buckets[key]
        ok_rows = [row for row in rows if row.get("status") == "ok"]
        accuracy_rows = _accuracy_denominator_rows(rows)
        correct = sum(1 for row in ok_rows if row.get("is_correct"))
        results.append(
            {
                group_field: key,
                "total": len(rows),
                "ok": len(ok_rows),
                "correct": correct,
                "accuracy": _safe_accuracy(correct, len(accuracy_rows)),
                "coverage": _safe_accuracy(len(ok_rows), len(rows)),
                "api_error": sum(1 for row in rows if row.get("status") == "api_error"),
                "parse_error": sum(1 for row in rows if row.get("status") == "parse_error"),
                "prior_bias_rate": _prior_bias_rate(rows),
                "variant_strict_acc": _variant_strict_accuracy(rows),
            }
        )
    return results
