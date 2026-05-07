from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _safe_rate(value: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(value / total * 100.0, 4)


def _safe_score_average(rows: List[Dict[str, Any]], field_name: str) -> float:
    values = [float(row[field_name]) for row in rows if row.get(field_name) is not None]
    if not values:
        return 0.0
    return round(sum(values) / len(values) * 100.0, 4)


def _safe_bool_count(rows: List[Dict[str, Any]], field_name: str) -> int:
    return sum(1 for row in rows if bool(row.get(field_name)))


def summarize_judge_records(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    ok_rows = [row for row in rows if row.get("judge_status") == "ok"]
    summary = {
        "total_records": len(rows),
        "judge_ok_records": len(ok_rows),
        "judge_coverage": _safe_rate(len(ok_rows), len(rows)),
        "cd_score_avg": _safe_score_average(ok_rows, "cd_score"),
        "rule_exact_match_count": _safe_bool_count(ok_rows, "rule_exact_match"),
        "rule_exact_match_rate": _safe_rate(_safe_bool_count(ok_rows, "rule_exact_match"), len(ok_rows)),
        "rule_conflict_detected_count": _safe_bool_count(ok_rows, "rule_conflict_detected"),
        "rule_conflict_detected_rate": _safe_rate(_safe_bool_count(ok_rows, "rule_conflict_detected"), len(ok_rows)),
        "quadrant_A": sum(1 for row in ok_rows if row.get("quadrant") == "A"),
        "quadrant_B": sum(1 for row in ok_rows if row.get("quadrant") == "B"),
        "quadrant_C": sum(1 for row in ok_rows if row.get("quadrant") == "C"),
        "quadrant_D": sum(1 for row in ok_rows if row.get("quadrant") == "D"),
        "type_i": sum(1 for row in ok_rows if row.get("error_type") == "type_i"),
        "type_ii": sum(1 for row in ok_rows if row.get("error_type") == "type_ii"),
        "undetermined": sum(1 for row in ok_rows if row.get("error_type") == "undetermined"),
        "none": sum(1 for row in ok_rows if row.get("error_type") == "none"),
    }
    return summary


def group_judge_metrics(records: Iterable[Dict[str, Any]], group_field: str) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in records:
        key = str(row.get(group_field, ""))
        buckets.setdefault(key, []).append(row)

    results: List[Dict[str, Any]] = []
    for key in sorted(buckets):
        rows = buckets[key]
        ok_rows = [row for row in rows if row.get("judge_status") == "ok"]
        results.append(
            {
                group_field: key,
                "total_records": len(rows),
                "judge_ok_records": len(ok_rows),
                "judge_coverage": _safe_rate(len(ok_rows), len(rows)),
                "cd_score_avg": _safe_score_average(ok_rows, "cd_score"),
                "rule_exact_match_count": _safe_bool_count(ok_rows, "rule_exact_match"),
                "rule_exact_match_rate": _safe_rate(_safe_bool_count(ok_rows, "rule_exact_match"), len(ok_rows)),
                "rule_conflict_detected_count": _safe_bool_count(ok_rows, "rule_conflict_detected"),
                "rule_conflict_detected_rate": _safe_rate(
                    _safe_bool_count(ok_rows, "rule_conflict_detected"), len(ok_rows)
                ),
                "quadrant_A": sum(1 for row in ok_rows if row.get("quadrant") == "A"),
                "quadrant_B": sum(1 for row in ok_rows if row.get("quadrant") == "B"),
                "quadrant_C": sum(1 for row in ok_rows if row.get("quadrant") == "C"),
                "quadrant_D": sum(1 for row in ok_rows if row.get("quadrant") == "D"),
                "type_i": sum(1 for row in ok_rows if row.get("error_type") == "type_i"),
                "type_ii": sum(1 for row in ok_rows if row.get("error_type") == "type_ii"),
                "undetermined": sum(1 for row in ok_rows if row.get("error_type") == "undetermined"),
                "none": sum(1 for row in ok_rows if row.get("error_type") == "none"),
            }
        )
    return results
