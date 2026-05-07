from __future__ import annotations

from typing import List, Tuple

from testCausal.sqa_stage2_dataset import Stage2Sample


PROMPT_MODES = {"answer_only_extractable", "direct_mcq", "cot_mcq"}


def format_choices(sample: Stage2Sample) -> str:
    return "\n".join(
        f"{label}. {choice}" for label, choice in zip(sample.option_labels, sample.choices)
    )


def build_system_message(prompt_mode: str) -> str:
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported prompt mode: {prompt_mode}")

    base = (
        "Use the attached image for the paired question only. "
        "Select exactly one option."
    )
    if prompt_mode == "cot_mcq":
        return (
            f"{base} Briefly reason in 1-3 sentences, then end with a final line formatted exactly as "
            "'Answer: X' where X is the option letter."
        )
    if prompt_mode == "direct_mcq":
        return (
            f"{base} Keep the response concise and end with a final line formatted exactly as "
            "'Answer: X' where X is the option letter."
        )
    return (
        f"{base} Return only the final answer line formatted exactly as "
        "'Answer: X' where X is the option letter."
    )


def _build_context_lines(sample: Stage2Sample) -> List[str]:
    context_lines: List[str] = [
        "Task: Solve the multiple-choice question using the attached image.",
    ]
    context_lines.append(f"Question: {sample.question}")
    context_lines.append("Options:")
    context_lines.append(format_choices(sample))
    return context_lines


def build_prompt(sample: Stage2Sample, prompt_mode: str = "answer_only_extractable") -> Tuple[str, str]:
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported prompt mode: {prompt_mode}")

    lines = _build_context_lines(sample)
    if prompt_mode == "cot_mcq":
        lines.append("Think carefully about the visible evidence and the causal relation asked in the question.")
        lines.append("Respond with a short explanation, then a final line in the format 'Answer: X'.")
    elif prompt_mode == "direct_mcq":
        lines.append("Choose the best option and end with a final line in the format 'Answer: X'.")
    else:
        lines.append("Return only one line in the format 'Answer: X'.")

    return "\n".join(lines).strip(), build_system_message(prompt_mode)
