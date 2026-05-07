from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _safe_accuracy(correct: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(correct / total * 100.0, 4)


def _accuracy_denominator_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("status") in {"ok", "parse_error"}]


def _non_empty_group_rows(rows: List[Dict[str, Any]], group_field: str) -> List[Dict[str, Any]]:
    return [row for row in rows if str(row.get(group_field, "")).strip()]


def _optional_group_metrics(rows: List[Dict[str, Any]], group_field: str) -> Dict[str, Dict[str, Any]]:
    filtered_rows = _non_empty_group_rows(rows, group_field)
    if not filtered_rows:
        return {}
    return {
        str(row[group_field]): row
        for row in group_metrics(filtered_rows, group_field)
    }


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
        "status_counts": status_counts,
        "text_match_fallback_hits": sum(1 for row in ok_rows if row.get("prediction_source") == "choice_text"),
        "letter_parse_hits": sum(1 for row in ok_rows if row.get("prediction_source") == "option_letter"),
        "samples_with_choice_images": sum(1 for row in rows if row.get("has_choice_images")),
    }
    stage2_quality_action_metrics = _optional_group_metrics(rows, "stage2_quality_action")
    if stage2_quality_action_metrics:
        summary["stage2_quality_action_metrics"] = stage2_quality_action_metrics
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
            }
        )
    return results
