from __future__ import annotations

from typing import Tuple

from test_visual_causal.vci_eval_dataset import VciSample


PROMPT_MODES = {
    "answer_only_extractable",
    "direct_mcq",
    "cot_mcq",
    "ca_cot",
    "ca_cot_zero_shot",
    "ca_cot_zero_shot_2step",
}

VCI_COT_FEW_SHOT_EXAMPLES = [
    {
        "question": "According to the pattern in the image, what would happen if the ramp angle became larger?",
        "visual_context": "The red arrow shows speed. Longer means faster.",
        "required_induction": "In this world, a larger ramp angle leads to a lower final speed.",
        "choices": [
            "The object moves faster.",
            "The object moves more slowly.",
            "The speed becomes the cause of angle.",
            "Nothing changes.",
        ],
        "step_1": "Across the three panels, the ramp becomes steeper while the red speed arrow becomes shorter.",
        "step_2": "The visual pattern shows that increasing ramp angle reduces final speed in this world.",
        "step_3": "So if the ramp angle becomes larger again, the object should end up moving more slowly.",
        "answer": "B",
    },
    {
        "question": "Which ramp should you choose if you want the fastest final speed?",
        "visual_context": "The red arrow shows speed. Longer means faster.",
        "required_induction": "In this world, a smaller ramp angle gives a higher final speed.",
        "choices": [
            "The steepest ramp.",
            "Any ramp works.",
            "The gentlest ramp.",
            "Speed decides the ramp angle.",
        ],
        "step_1": "The gentlest ramp is paired with the longest speed arrow, and steeper ramps are paired with shorter arrows.",
        "step_2": "The image implies that smaller angle means higher final speed in this world.",
        "step_3": "To maximize final speed, you should therefore choose the gentlest ramp.",
        "answer": "C",
    },
]


def is_vci_cot_prompt_mode(prompt_mode: str) -> bool:
    return prompt_mode in {"ca_cot", "ca_cot_zero_shot", "ca_cot_zero_shot_2step"}


def get_vci_cot_required_steps(prompt_mode: str) -> tuple[int, ...]:
    if prompt_mode == "ca_cot_zero_shot_2step":
        return (1, 2)
    if is_vci_cot_prompt_mode(prompt_mode):
        return (1, 2, 3, 4)
    return ()


def format_choices(sample: VciSample) -> str:
    return "\n".join(
        f"{label}. {choice}" for label, choice in zip(sample.option_labels, sample.choices)
    )


def build_question_with_context(question: str, context_text: str, context_label: str) -> str:
    normalized_question = question.strip()
    normalized_context = context_text.strip()
    if not normalized_context:
        return normalized_question
    return f"{context_label}: {normalized_context}\n{normalized_question}"


def format_frame_descriptions(frame_descriptions: dict[str, str]) -> str:
    if not frame_descriptions:
        return ""

    def sort_key(item: tuple[str, str]) -> tuple[int, str]:
        key, _ = item
        suffix = key.rsplit("_", 1)[-1]
        if suffix.isdigit():
            return int(suffix), key
        return 10_000, key

    lines = ["Frame descriptions:"]
    for key, description in sorted(frame_descriptions.items(), key=sort_key):
        normalized_key = key.replace("_", " ").strip().title()
        lines.append(f"{normalized_key}: {description}")
    return "\n".join(lines)


def resolve_question_context(
    sample: VciSample,
    input_modality: str,
    text_context_source: str,
) -> Tuple[str, str]:
    if input_modality == "text_only":
        if text_context_source == "visual_context" and sample.visual_context.strip():
            return sample.visual_context.strip(), "Text context"
        if text_context_source == "frame_descriptions":
            context_parts: list[str] = []
            if sample.visual_context.strip():
                context_parts.append(f"Visual context: {sample.visual_context.strip()}")
            frame_description_text = format_frame_descriptions(sample.frame_descriptions)
            if frame_description_text:
                context_parts.append(frame_description_text)
            return "\n".join(context_parts), "Text context"
        return "", "Text context"
    return sample.visual_context.strip(), "Visual context"


def build_system_message(prompt_mode: str, input_modality: str = "multimodal") -> str:
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported prompt mode: {prompt_mode}")

    if input_modality == "text_only":
        base = (
            "Use the question and any provided text context to answer the paired question. "
            "Select exactly one option."
        )
    else:
        base = (
            "Select exactly one option."
        )
    if is_vci_cot_prompt_mode(prompt_mode):
        if prompt_mode == "ca_cot_zero_shot_2step":
            return (
                f"{base} Follow exactly two labeled steps: 'Step 1: Rule Induction' and "
                f"'Step 2: Final Answer'. In Step 1, state the induced rule from the image. "
                f"In Step 2, write only 'Answer: X' where X is the option letter."
            )
        return (
            f"{base} Follow exactly four labeled steps: 'Step 1: Visual Observation', "
            f"'Step 2: Rule Induction', 'Step 3: Reasoning Process', and 'Step 4: Final Answer'. "
            f"In Step 1, report only the provided evidence. In Step 2, state the induced rule from that evidence. "
            f"In Step 3, show the reasoning process that leads to the answer. In Step 4, write only 'Answer: X' "
            f"where X is the option letter."
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


def _format_labeled_choices(choices: list[str]) -> str:
    return "\n".join(
        f"{label}. {choice}" for label, choice in zip([chr(code) for code in range(ord('A'), ord('A') + len(choices))], choices)
    )


def _build_vci_cot_example_block(example: dict[str, object], example_index: int) -> str:
    lines = [
        f"Worked Example {example_index}",
        f"Question: {build_question_with_context(str(example['question']), str(example['visual_context']), 'Visual context')}",
        "Options:",
        _format_labeled_choices(example["choices"]),
        "Assistant response:",
        "Step 1: Visual Observation",
        str(example["step_1"]),
        "Step 2: Rule Induction",
        str(example["step_2"]),
        "Step 3: Reasoning Process",
        str(example["step_3"]),
        "Step 4: Final Answer",
        f"Answer: {example['answer']}",
    ]
    return "\n".join(lines)


def _build_vci_cot_prompt(
    sample: VciSample,
    prompt_mode: str,
    use_few_shot: bool,
    input_modality: str,
    text_context_source: str,
) -> str:
    sections: list[str] = []
    if use_few_shot:
        sections.append("Use the worked examples to follow the VCI-CoT format.")
        sections.extend(
            _build_vci_cot_example_block(example, example_index)
            for example_index, example in enumerate(VCI_COT_FEW_SHOT_EXAMPLES, start=1)
        )

    context_text, context_label = resolve_question_context(sample, input_modality, text_context_source)
    target_lines: list[str] = []
    target_lines.extend(
        [
            f"Question: {build_question_with_context(sample.question, context_text, context_label)}",
            "Options:",
            format_choices(sample),
            "Now solve the target question.",
            "Follow this format exactly:",
        ]
    )
    if prompt_mode == "ca_cot_zero_shot_2step":
        target_lines.extend(
            [
                "Step 1: Rule Induction",
                "State the induced rule from the image.",
                "Step 2: Final Answer",
                "Answer: X",
            ]
        )
    else:
        target_lines.extend(
            [
                "Step 1: Visual Observation",
                "Describe the visible trend in the image.",
                "Step 2: Rule Induction",
                "State the induced rule from the image.",
                "Step 3: Reasoning Process",
                "Show the reasoning process that leads to the answer.",
                "Step 4: Final Answer",
                "Answer: X",
            ]
        )
    sections.append("\n".join(target_lines))
    return "\n\n".join(sections).strip()


def build_prompt(
    sample: VciSample,
    prompt_mode: str = "answer_only_extractable",
    input_modality: str = "multimodal",
    text_context_source: str = "visual_context",
) -> Tuple[str, str]:
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported prompt mode: {prompt_mode}")

    if is_vci_cot_prompt_mode(prompt_mode):
        return (
            _build_vci_cot_prompt(
                sample,
                prompt_mode=prompt_mode,
                use_few_shot=(prompt_mode == "ca_cot"),
                input_modality=input_modality,
                text_context_source=text_context_source,
            ),
            build_system_message(prompt_mode, input_modality=input_modality),
        )

    context_text, context_label = resolve_question_context(sample, input_modality, text_context_source)
    lines: list[str] = []
    lines.append(f"Question: {build_question_with_context(sample.question, context_text, context_label)}")
    lines.append("Options:")
    lines.append(format_choices(sample))

    if prompt_mode == "cot_mcq":
        lines.append("Reason briefly from the visible trend and then answer.")
        lines.append("End with a final line in the format 'Answer: X'.")
    elif prompt_mode == "direct_mcq":
        lines.append("Choose the best option and end with a final line in the format 'Answer: X'.")
    else:
        lines.append("Return only one line in the format 'Answer: X'.")

    return "\n".join(lines).strip(), build_system_message(prompt_mode, input_modality=input_modality)
