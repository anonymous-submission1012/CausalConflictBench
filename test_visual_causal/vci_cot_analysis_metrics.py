from __future__ import annotations

from typing import Any, Dict, Iterable, List


JOINT_BUCKETS = (
    "cot_correct_answer_correct",
    "cot_correct_answer_wrong",
    "cot_wrong_answer_correct",
    "cot_wrong_answer_wrong",
    "cot_undecidable",
)
RULE_SOURCE_BUCKETS = ("factual_rule", "flipped_rule", "other")


def _safe_accuracy(correct: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(correct / total * 100.0, 4)


def summarize_records(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    total = len(rows)
    ok_rows = [row for row in rows if row.get("analysis_status") == "ok"]
    judgeable_rows = [row for row in ok_rows if row.get("cot_rule_label") in {"correct", "incorrect"}]
    correct_rows = [row for row in judgeable_rows if row.get("cot_rule_label") == "correct"]
    source_labeled_rows = [row for row in ok_rows if row.get("cot_rule_source_label") in RULE_SOURCE_BUCKETS]

    status_counts: Dict[str, int] = {}
    for row in rows:
        status = str(row.get("analysis_status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    summary: Dict[str, Any] = {
        "total_records": total,
        "analysis_ok_records": len(ok_rows),
        "analysis_coverage": _safe_accuracy(len(ok_rows), total),
        "cot_rule_judgeable_records": len(judgeable_rows),
        "cot_rule_correct_records": len(correct_rows),
        "cot_rule_accuracy": _safe_accuracy(len(correct_rows), len(judgeable_rows)),
        "status_counts": status_counts,
        "answer_correct_rate_given_cot_correct": _safe_accuracy(
            sum(1 for row in ok_rows if row.get("joint_bucket") == "cot_correct_answer_correct"),
            sum(1 for row in ok_rows if row.get("joint_bucket") in {"cot_correct_answer_correct", "cot_correct_answer_wrong"}),
        ),
        "answer_correct_rate_given_cot_wrong": _safe_accuracy(
            sum(1 for row in ok_rows if row.get("joint_bucket") == "cot_wrong_answer_correct"),
            sum(1 for row in ok_rows if row.get("joint_bucket") in {"cot_wrong_answer_correct", "cot_wrong_answer_wrong"}),
        ),
        "answer_correct_rate_given_factual_rule": _safe_accuracy(
            sum(1 for row in source_labeled_rows if row.get("cot_rule_source_label") == "factual_rule" and row.get("is_correct")),
            sum(1 for row in source_labeled_rows if row.get("cot_rule_source_label") == "factual_rule"),
        ),
        "answer_correct_rate_given_flipped_rule": _safe_accuracy(
            sum(1 for row in source_labeled_rows if row.get("cot_rule_source_label") == "flipped_rule" and row.get("is_correct")),
            sum(1 for row in source_labeled_rows if row.get("cot_rule_source_label") == "flipped_rule"),
        ),
        "cot_rule_source_factual_rule_ratio": _safe_accuracy(
            sum(1 for row in source_labeled_rows if row.get("cot_rule_source_label") == "factual_rule"),
            len(source_labeled_rows),
        ),
    }
    for bucket in JOINT_BUCKETS:
        summary[bucket] = sum(1 for row in ok_rows if row.get("joint_bucket") == bucket)
    for bucket in RULE_SOURCE_BUCKETS:
        summary[f"cot_rule_source_{bucket}"] = sum(1 for row in source_labeled_rows if row.get("cot_rule_source_label") == bucket)
    return summary


def group_metrics(records: Iterable[Dict[str, Any]], group_field: str) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in records:
        key = str(row.get(group_field, ""))
        buckets.setdefault(key, []).append(row)

    grouped_rows: List[Dict[str, Any]] = []
    for key in sorted(buckets):
        group_rows = buckets[key]
        summary = summarize_records(group_rows)
        grouped_row = {
            group_field: key,
            "total_records": summary["total_records"],
            "analysis_ok_records": summary["analysis_ok_records"],
            "analysis_coverage": summary["analysis_coverage"],
            "cot_rule_judgeable_records": summary["cot_rule_judgeable_records"],
            "cot_rule_correct_records": summary["cot_rule_correct_records"],
            "cot_rule_accuracy": summary["cot_rule_accuracy"],
            "answer_correct_rate_given_factual_rule": summary["answer_correct_rate_given_factual_rule"],
            "answer_correct_rate_given_flipped_rule": summary["answer_correct_rate_given_flipped_rule"],
            "cot_rule_source_factual_rule_ratio": summary["cot_rule_source_factual_rule_ratio"],
        }
        for bucket in JOINT_BUCKETS:
            grouped_row[bucket] = summary[bucket]
        for bucket in RULE_SOURCE_BUCKETS:
            grouped_row[f"cot_rule_source_{bucket}"] = summary[f"cot_rule_source_{bucket}"]
        grouped_rows.append(grouped_row)
    return grouped_rows
