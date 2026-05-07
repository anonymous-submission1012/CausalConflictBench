from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _safe_accuracy(correct: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(correct / total * 100.0, 4)


def _accuracy_denominator_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("status") in {"ok", "parse_error"}]


def _baseline_bias_rate(rows: List[Dict[str, Any]]) -> float:
    wrong_ok_rows = [row for row in rows if row.get("status") == "ok" and not row.get("is_correct")]
    baseline_hits = sum(1 for row in wrong_ok_rows if row.get("matches_factual_baseline"))
    return _safe_accuracy(baseline_hits, len(wrong_ok_rows))


def _supports_baseline_bias(rows: List[Dict[str, Any]]) -> bool:
    conflict_rows = [row for row in rows if str(row.get("task_variant", "conflict")) == "conflict"]
    return bool(conflict_rows)


def _qid_strict_accuracy(rows: List[Dict[str, Any]]) -> float:
    rows_with_qid = [row for row in rows if str(row.get("qid", "")).strip()]
    if not rows_with_qid:
        return 0.0

    grouped_rows: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows_with_qid:
        grouped_rows.setdefault(str(row.get("qid")), []).append(row)

    strict_correct = 0
    for qid_rows in grouped_rows.values():
        if all(row.get("status") == "ok" and bool(row.get("is_correct")) for row in qid_rows):
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

    summary = {
        "total_records": total,
        "ok_records": len(ok_rows),
        "correct_records": correct,
        "accuracy": _safe_accuracy(correct, len(accuracy_rows)),
        "coverage": _safe_accuracy(len(ok_rows), total),
        "qid_acc": _qid_strict_accuracy(rows),
        "status_counts": status_counts,
        "text_match_fallback_hits": sum(1 for row in ok_rows if row.get("prediction_source") == "choice_text"),
        "letter_parse_hits": sum(1 for row in ok_rows if row.get("prediction_source") == "option_letter"),
    }
    if _supports_baseline_bias(rows):
        summary["factual_baseline_bias_rate"] = _baseline_bias_rate(rows)
    return summary


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
        result = {
            group_field: key,
            "total": len(rows),
            "ok": len(ok_rows),
            "correct": correct,
            "accuracy": _safe_accuracy(correct, len(accuracy_rows)),
            "coverage": _safe_accuracy(len(ok_rows), len(rows)),
            "api_error": sum(1 for row in rows if row.get("status") == "api_error"),
            "parse_error": sum(1 for row in rows if row.get("status") == "parse_error"),
        }
        if _supports_baseline_bias(rows):
            result["factual_baseline_bias_rate"] = _baseline_bias_rate(rows)
        results.append(result)
    return results


def group_metrics_by_fields(records: Iterable[Dict[str, Any]], group_fields: List[str]) -> List[Dict[str, Any]]:
    buckets: Dict[tuple[str, ...], List[Dict[str, Any]]] = {}
    for row in records:
        key = tuple(str(row.get(group_field, "")) for group_field in group_fields)
        buckets.setdefault(key, []).append(row)

    results: List[Dict[str, Any]] = []
    for key in sorted(buckets):
        rows = buckets[key]
        ok_rows = [row for row in rows if row.get("status") == "ok"]
        accuracy_rows = _accuracy_denominator_rows(rows)
        correct = sum(1 for row in ok_rows if row.get("is_correct"))
        result: Dict[str, Any] = {group_field: key[index] for index, group_field in enumerate(group_fields)}
        result.update(
            {
                "total": len(rows),
                "ok": len(ok_rows),
                "correct": correct,
                "accuracy": _safe_accuracy(correct, len(accuracy_rows)),
                "coverage": _safe_accuracy(len(ok_rows), len(rows)),
                "api_error": sum(1 for row in rows if row.get("status") == "api_error"),
                "parse_error": sum(1 for row in rows if row.get("status") == "parse_error"),
            }
        )
        if _supports_baseline_bias(rows):
            result["factual_baseline_bias_rate"] = _baseline_bias_rate(rows)
        results.append(result)
    return results
