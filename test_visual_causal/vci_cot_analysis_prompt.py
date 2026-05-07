from __future__ import annotations

import json
from typing import Any, Dict, Tuple


FLIP_RULE_LABELS = {"correct", "incorrect", "undecidable"}
RULE_SOURCE_LABELS = {"factual_rule", "flipped_rule", "other"}
JUDGE_LABELS = FLIP_RULE_LABELS


FLIP_RULE_FEW_SHOT_EXAMPLES = [
    {
        "step_1": "As the light becomes stronger, the pupil gets larger.",
        "required_induction": "In this world, stronger light leads to a larger pupil.",
        "label": "correct",
    },
    {
        "step_1": "As heating increases, the metal ball becomes larger, so it passes through the ring less easily.",
        "required_induction": "In this world, increasing the metal ball's temperature makes it smaller, so it passes through the fixed ring more easily.",
        "label": "incorrect",
    },
    {
        "step_1": "The left panel is darker and the right panel is brighter.",
        "required_induction": "In this world, brighter light leads to a larger pupil.",
        "label": "undecidable",
    },
]


RULE_SOURCE_FEW_SHOT_EXAMPLES = [
    {
        "question": "According to the pattern in the image, what happens when the light becomes stronger?",
        "factual_rule": "环境光照强度↑ → 瞳孔缩小↓",
        "flipped_rule": "环境光照强度↑ → 瞳孔扩大↑",
        "step_1": "In the real world, stronger light leads to a smaller pupil.",
        "required_induction": "In this world, stronger light leads to a larger pupil.",
        "label": "factual_rule",
    },
    {
        "question": "According to the pattern in the image, what happens when the light becomes weaker?",
        "factual_rule": "环境光照强度↑ → 瞳孔缩小↓",
        "flipped_rule": "环境光照强度↑ → 瞳孔扩大↑",
        "step_1": "In this world, weaker light leads to a smaller pupil.",
        "required_induction": "In this world, weaker light leads to a smaller pupil.",
        "label": "flipped_rule",
    },
    {
        "question": "According to the pattern in the image, what happens when the light becomes stronger?",
        "factual_rule": "环境光照强度↑ → 瞳孔缩小↓",
        "flipped_rule": "环境光照强度↑ → 瞳孔扩大↑",
        "step_1": "I will choose the answer saying the pupil gets larger.",
        "required_induction": "In this world, stronger light leads to a larger pupil.",
        "label": "other",
    },
]


BOTH_MODE_FEW_SHOT_EXAMPLES = [
    {
        "question": "According to the pattern in the image, what happens when the light becomes stronger?",
        "factual_rule": "环境光照强度↑ → 瞳孔缩小↓",
        "flipped_rule": "环境光照强度↑ → 瞳孔扩大↑",
        "step_1": "In this world, stronger light leads to a larger pupil.",
        "required_induction": "In this world, stronger light leads to a larger pupil.",
        "cot_rule_label": "correct",
        "cot_rule_source_label": "flipped_rule",
    },
    {
        "question": "According to the pattern in the image, what happens when the light becomes stronger?",
        "factual_rule": "环境光照强度↑ → 瞳孔缩小↓",
        "flipped_rule": "环境光照强度↑ → 瞳孔扩大↑",
        "step_1": "In the real world, stronger light leads to a smaller pupil.",
        "required_induction": "In this world, stronger light leads to a larger pupil.",
        "cot_rule_label": "incorrect",
        "cot_rule_source_label": "factual_rule",
    },
    {
        "question": "According to the pattern in the image, what happens when the light becomes stronger?",
        "factual_rule": "环境光照强度↑ → 瞳孔缩小↓",
        "flipped_rule": "环境光照强度↑ → 瞳孔扩大↑",
        "step_1": "I will choose the option saying the pupil gets larger.",
        "required_induction": "In this world, stronger light leads to a larger pupil.",
        "cot_rule_label": "undecidable",
        "cot_rule_source_label": "other",
    },
]


def build_system_message(analysis_mode: str = "flip_rule") -> str:
    if analysis_mode == "rule_source":
        return (
            "Decide which rule source the model's Step 1 expresses under the direction asked by the current question: "
            "the factual_rule field, the flipped_rule field, or neither. "
            "Use the rule-to-match field as the directional anchor for the current question. "
            "Label factual_rule only if Step 1 follows the factual_rule relation in the current question direction. "
            "Label flipped_rule if Step 1 follows the flipped_rule relation in the current question direction, "
            "including cases where Step 1 states the inverse direction of the flipped_rule field. "
            "Do not label factual_rule merely because Step 1 shares an outcome phrase with factual_rule. "
            "Use other if Step 1 is answer-choice matching, incomplete, mixed, or unclear. "
            "Ignore answer-choice text. Judge only what rule the Step 1 is expressing. "
            "Return JSON only in the format "
            "{\"cot_rule_source_label\":\"factual_rule|flipped_rule|other\"}."
        )
    if analysis_mode == "both":
        return (
            "Judge two things about the model's Step 1. "
            "First, whether it matches the rule-to-match field with cot_rule_label=correct|incorrect|undecidable. "
            "Second, decide which rule source Step 1 expresses under the direction asked by the current question: "
            "the factual_rule field, the flipped_rule field, or neither. "
            "Use the rule-to-match field as the directional anchor. "
            "Use factual_rule only if Step 1 follows the factual_rule relation in the current question direction. "
            "Use flipped_rule if Step 1 follows the flipped_rule relation in the current question direction, "
            "including inverse-direction expressions of the flipped_rule field. "
            "Use other for answer-choice matching, incomplete, mixed, or unclear rules. "
            "Ignore answer-choice text. Judge only what rule the Step 1 is expressing. "
            "Return JSON only in the format "
            "{\"cot_rule_label\":\"correct|incorrect|undecidable\",\"cot_rule_source_label\":\"factual_rule|flipped_rule|other\"}."
        )
    return (
        "Judge whether the model's Step 1 rule induction matches the reference rule. "
        "Referring to rules in this benchmark is generally counter common sense and different from the real world. "
        "Ignore answer-choice text. Judge only what rule the Step 1 is expressing. "
        "Return JSON only in the format {\"cot_rule_label\":\"correct|incorrect|undecidable\"}."
    )


def _render_output_payload(example: Dict[str, Any], analysis_mode: str) -> str:
    if analysis_mode == "rule_source":
        return json.dumps({"cot_rule_source_label": example["label"]})
    if analysis_mode == "both":
        return json.dumps(
            {
                "cot_rule_label": example["cot_rule_label"],
                "cot_rule_source_label": example["cot_rule_source_label"],
            }
        )
    return json.dumps({"cot_rule_label": example["label"]})


def _get_few_shot_examples(analysis_mode: str) -> list[Dict[str, Any]]:
    if analysis_mode == "rule_source":
        return RULE_SOURCE_FEW_SHOT_EXAMPLES
    if analysis_mode == "both":
        return BOTH_MODE_FEW_SHOT_EXAMPLES
    return FLIP_RULE_FEW_SHOT_EXAMPLES


def build_prompt(prediction_row: Dict[str, Any], reference_row: Dict[str, Any], analysis_mode: str = "flip_rule") -> Tuple[str, str]:
    sections: list[str] = [
        "# Few-Shot Examples",
    ]
    for index, example in enumerate(_get_few_shot_examples(analysis_mode), start=1):
        example_lines = [f"Example {index}"]
        if "question" in example:
            example_lines.append(f"Question: {example['question']}")
        if "factual_rule" in example:
            example_lines.append(f"Factual rule: {example['factual_rule']}")
        if "flipped_rule" in example:
            example_lines.append(f"Flipped rule: {example['flipped_rule']}")
        example_lines.append(f"Model Step 1: {example['step_1']}")
        if "required_induction" in example:
            example_lines.append(f"Rule to match: {example['required_induction']}")
        example_lines.append(f"Output: {_render_output_payload(example, analysis_mode)}")
        sections.append(
            "\n".join(example_lines)
        )

    choice_lines = []
    for label, choice in sorted((reference_row.get("choices") or {}).items()):
        choice_lines.append(f"{label}. {choice}")
    choices_block = "\n".join(choice_lines) if choice_lines else "(choices unavailable)"

    sections.append(
        "\n".join(
            [
                "# Target Example",
                f"QID: {prediction_row.get('qid', '')}",
                f"Question: {reference_row.get('question', '')}",
                "Choices:",
                choices_block,
                f"Factual rule: {reference_row.get('factual_rule', '')}",
                f"Flipped rule: {reference_row.get('flipped_rule', '')}",
                f"Model Step 1: {prediction_row.get('vci_cot_step_1', '')}",
                *([f"Rule to match: {reference_row.get('required_induction', '')}"] if analysis_mode in {"both", "flip_rule", "rule_source"} else []),
                "",
                "Return JSON only.",
            ]
        )
    )
    return "\n\n".join(sections).strip(), build_system_message(analysis_mode=analysis_mode)
