from __future__ import annotations

import copy
from typing import Sequence, Tuple

from testCausal.sqa_stage3_conflict_dataset import Stage3ConflictSample


PROMPT_MODES = {
    "answer_only_extractable",
    "direct_mcq",
    "cot_mcq",
    "ca_cot",
    "ca_cot_zero_shot",
    "ca_cot_zero_shot_2step",
}
RULE_POSITIONS = {"prefix", "suffix"}
INPUT_MODALITIES = {"multimodal", "text_only"}
TEXT_CONTEXT_SOURCES = {"caption", "none"}


def is_ca_cot_prompt_mode(prompt_mode: str) -> bool:
    return prompt_mode in {"ca_cot", "ca_cot_zero_shot", "ca_cot_zero_shot_2step"}


def get_ca_cot_required_steps(prompt_mode: str) -> tuple[int, ...]:
    if prompt_mode == "ca_cot_zero_shot_2step":
        return (1, 2)
    if is_ca_cot_prompt_mode(prompt_mode):
        return (1, 2, 3, 4)
    return ()

CA_COT_FEW_SHOT_EXAMPLES = [
    {
        "question": "Will these magnets attract or repel each other?",
        "premise": (
            "In this magnet model, opposite poles pull together only when the gap between the facing pole faces "
            "is less than one-third of a pole-label block width; beyond that distance, the magnetic pull is too "
            "weak to produce motion."
        ),
        "choices": [
            "attract more strongly because a wider gap increases the pull",
            "attract",
            "repel",
            "neither; they remain separated",
        ],
        "step_1": "The image shows an opposite-pole N-S facing pair with a visible white gap between the magnets.",
        "step_2": (
            "Ordinary magnet knowledge says opposite poles attract, but the given premise adds a strict "
            "distance threshold, so attraction does not always cause motion."
        ),
        "step_3": (
            "Because the visible gap is wider than the stated threshold, the attraction stays too weak to move "
            "the magnets. Under the given rule, they remain separated."
        ),
        "answer": "D",
    },
    {
        "question": "Compare the average kinetic energies of the particles in each sample. Which sample has the higher temperature?",
        "premise": "In this particle model, temperature decreases as the average kinetic energy of the particles increases.",
        "choices": [
            "neither; the samples have the same temperature",
            "sample A by an even larger margin",
            "sample A",
            "sample B",
        ],
        "step_1": (
            "The image shows two sealed rigid containers labeled sample A and sample B, each containing the same "
            "number of particles. In sample A, the green particles have larger mass (46 u) and higher average "
            "speed (1,100 m/s), with more motion lines around them. In sample B, the blue particles have smaller "
            "mass (32 u) and lower average speed (1,000 m/s), with fewer motion lines."
        ),
        "step_2": (
            "Normally higher average kinetic energy means higher temperature, but the given premise reverses "
            "that direction."
        ),
        "step_3": (
            "Since sample A has the greater average kinetic energy, the reversed rule makes sample A the colder "
            "sample. Therefore sample B has the higher temperature."
        ),
        "answer": "D",
    },
    {
        "question": "Based on the temperature data and the environmental conditions described, what is the direction of net thermal energy transfer for the cake that ended at the lower final temperature?",
        "premise": "The surrounding air is hotter than the cakes. For a cake exposed on a plate, evaporative cooling at the uncovered surface can significantly lower the thermometer reading, so a falling temperature does not necessarily indicate outward heat flow.",
        "choices": [
            "the cake to the surroundings, because stronger evaporative cooling must mean the cake is sending more heat outward",
            "the cake to the surroundings",
            "the surroundings to the cake",
            "the temperature drop was caused only by evaporation, so no heat-transfer direction applies",
        ],
        "step_1": (
            "The table compares the temperature changes of two cakes placed in different conditions. Both cakes "
            "start at 92 degrees C. After some time, the cake left in the pan ends at 82 degrees C, while the "
            "cake moved onto a plate ends at the lower temperature of 79 degrees C."
        ),
        "step_2": (
            "Although temperature decreases are typically taken as evidence that heat is leaving an object, the "
            "premise introduces evaporative cooling, which can reduce the measured temperature independently of "
            "net heat flow direction."
        ),
        "step_3": (
            "Since the surrounding air is hotter than the cake, heat flows from the surroundings into the cake. "
            "The observed temperature drop is driven by surface evaporation, not by net outward heat transfer. "
            "Therefore, thermal energy was transferred from the surroundings to the cake."
        ),
        "answer": "C",
    },
]


def get_ca_cot_few_shot_examples() -> list[dict[str, object]]:
    return copy.deepcopy(CA_COT_FEW_SHOT_EXAMPLES)


def render_ca_cot_few_shot_examples_markdown() -> str:
    lines = ["# CA-CoT Few-Shot Examples", ""]
    for index, example in enumerate(CA_COT_FEW_SHOT_EXAMPLES, start=1):
        lines.extend(
            [
                f"## Example {index}",
                f"- Question: {example['question']}",
                f"- Premise: {example['premise']}",
                "",
                "Options:",
                format_labeled_choices(
                    [chr(code) for code in range(ord("A"), ord("A") + len(example["choices"]))],
                    example["choices"],
                ),
                "",
                "Step 1: Visual Observation",
                str(example["step_1"]),
                "",
                "Step 2: Conflict Detection",
                str(example["step_2"]),
                "",
                "Step 3: Reasoning",
                str(example["step_3"]),
                "",
                "Step 4: Final Answer",
                f"Answer: {example['answer']}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def format_choices(sample: Stage3ConflictSample) -> str:
    return format_labeled_choices(sample.option_labels, sample.choices)


def format_labeled_choices(option_labels: Sequence[str], choices: Sequence[str]) -> str:
    return "\n".join(f"{label}. {choice}" for label, choice in zip(option_labels, choices))


def build_system_message(
    prompt_mode: str,
    input_modality: str = "multimodal",
    text_context_source: str = "caption",
) -> str:
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported prompt mode: {prompt_mode}")
    if input_modality not in INPUT_MODALITIES:
        raise ValueError(f"Unsupported input_modality: {input_modality}")
    if text_context_source not in TEXT_CONTEXT_SOURCES:
        raise ValueError(f"Unsupported text_context_source: {text_context_source}")

    if input_modality == "multimodal":
        base = (
            "Use the attached image for the paired question only. "
            "Select exactly one option."
        )
    else:
        context_phrase = "Use the provided text description and question only."
        if text_context_source == "none":
            context_phrase = "Use the provided question and options only."
        base = f"You are evaluating a text-only benchmark. {context_phrase} Select exactly one option."
    if is_ca_cot_prompt_mode(prompt_mode):
        if prompt_mode == "ca_cot_zero_shot_2step":
            return (
                f"{base} Follow exactly two labeled steps: 'Step 1: Rule recognition' and "
                f"'Step 2: Final Answer'. In Step 1, summarize the rules required for solving "
                f"problems based on graphic and textual information. In Step 2, write only "
                f"'Answer: X' where X is the option letter."
            )
        return (
            f"{base} Follow exactly four labeled steps: 'Step 1: Visual Observation', "
            f"'Step 2: Rule recognition', 'Step 3: Reasoning', and "
            f"'Step 4: Final Answer'. In Step 1, report only visible evidence. In Step 2, summarize the rules required for solving problems based on graphic and textual information"
            f". In Step 3, briefly describe the reasoning process"
            f". In Step 4, write only 'Answer: X' where X is the option letter."
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
    return f"{base} Return only the final answer line formatted exactly as 'Answer: X' where X is the option letter."


def _extract_question_text(sample: Stage3ConflictSample) -> str:
    if sample.task_variant == "conflict" and "Question:" in sample.question:
        return sample.question.split("Question:", 1)[1].strip()
    return sample.question.strip()


def _build_shared_prompt_lines(
    sample: Stage3ConflictSample,
    rule_position: str,
    input_modality: str,
    text_context_source: str,
) -> list[str]:
    task_line = "Task: Solve the question using the attached image."
    if input_modality == "text_only":
        task_line = "Task: Solve the question using the provided text description."
        if text_context_source == "none":
            task_line = "Task: Solve the question using only the provided question and options."
    lines = [f"Question ID: {sample.sample_id}", task_line]
    if input_modality == "text_only" and text_context_source == "caption" and sample.image_caption:
        lines.append(f"Image description: {sample.image_caption}")
    question_text = _extract_question_text(sample)
    if rule_position == "prefix":
        if sample.task_variant == "distractor":
            lines.extend([f"Premise: {sample.flipped_rule}", f"Question: {question_text}"])
        else:
            lines.append(f"{sample.question}")
        lines.extend(["Options:", format_choices(sample)])
    else:
        lines.extend(
            [
                f"Question: {question_text}",
                f"Premise: {sample.flipped_rule}",
                "Options:",
                format_choices(sample),
            ]
        )
    return lines


def _build_ca_cot_example_block(example: dict[str, object], example_index: int) -> str:
    choices = example["choices"]
    option_labels = [chr(code) for code in range(ord("A"), ord("A") + len(choices))]
    lines = [
        f"Worked Example {example_index}",
        f"Question: {example['question']}",
        f"Premise: {example['premise']}",
        "Options:",
        format_labeled_choices(option_labels, choices),
        "Assistant response:",
        "Step 1: Visual Observation",
        str(example["step_1"]),
        "Step 2: Rule recognition",
        str(example["step_2"]),
        "Step 3: Reasoning",
        str(example["step_3"]),
        "Step 4: Final Answer",
        f"Answer: {example['answer']}",
    ]
    return "\n".join(lines)


def _build_ca_cot_prompt(
    sample: Stage3ConflictSample,
    prompt_mode: str,
    rule_position: str,
    input_modality: str,
    text_context_source: str,
    use_few_shot: bool,
) -> str:
    example_blocks = []
    if use_few_shot:
        example_blocks = [
            _build_ca_cot_example_block(example, example_index)
            for example_index, example in enumerate(CA_COT_FEW_SHOT_EXAMPLES, start=1)
        ]
    target_lines = _build_shared_prompt_lines(
        sample,
        rule_position=rule_position,
        input_modality=input_modality,
        text_context_source=text_context_source,
    )
    target_lines.extend(["Now solve the target question.", "Follow this format exactly:"])
    if prompt_mode == "ca_cot_zero_shot_2step":
        target_lines.extend(
            [
                "Step 1: Rule recognition",
                "summarize the rules required for solving problems based on graphic and textual information.",
                "Step 2: Final Answer",
                "Answer: X",
            ]
        )
    else:
        target_lines.extend(
            [
                "Step 1: Visual Observation",
                "Describe the image content.",
                "Step 2: Rule recognition",
                "summarize the rules required for solving problems based on graphic and textual information.",
                "Step 3: Reasoning",
                "Briefly describe the reasoning process.",
                "Step 4: Final Answer",
                "Answer: X",
            ]
        )
    sections = []
    if use_few_shot:
        sections.extend(
            [
                "Use the worked examples to follow the CA-CoT format.",
                *example_blocks,
            ]
        )
    sections.append("\n".join(target_lines))
    return "\n\n".join(sections).strip()


def build_prompt(
    sample: Stage3ConflictSample,
    prompt_mode: str = "answer_only_extractable",
    rule_position: str = "prefix",
    input_modality: str = "multimodal",
    text_context_source: str = "caption",
) -> Tuple[str, str]:
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported prompt mode: {prompt_mode}")
    if rule_position not in RULE_POSITIONS:
        raise ValueError(f"Unsupported rule_position: {rule_position}")
    if input_modality not in INPUT_MODALITIES:
        raise ValueError(f"Unsupported input_modality: {input_modality}")
    if text_context_source not in TEXT_CONTEXT_SOURCES:
        raise ValueError(f"Unsupported text_context_source: {text_context_source}")

    if is_ca_cot_prompt_mode(prompt_mode):
        return _build_ca_cot_prompt(
            sample,
            prompt_mode=prompt_mode,
            rule_position=rule_position,
            input_modality=input_modality,
            text_context_source=text_context_source,
            use_few_shot=(prompt_mode == "ca_cot"),
        ), build_system_message(
            prompt_mode,
            input_modality=input_modality,
            text_context_source=text_context_source,
        )

    lines = _build_shared_prompt_lines(
        sample,
        rule_position=rule_position,
        input_modality=input_modality,
        text_context_source=text_context_source,
    )
    if prompt_mode == "cot_mcq":
        if sample.task_variant == "distractor":
            lines.append("Briefly explain the visible evidence before giving the answer.")
        else:
            lines.append("Briefly explain the visible evidence and how the premise changes the answer.")
        lines.append("Finish with a final line in the format 'Answer: X'.")
    elif prompt_mode == "direct_mcq":
        lines.append("Choose the best option and end with a final line in the format 'Answer: X'.")
    else:
        lines.append("Return only one line in the format 'Answer: X'.")

    return "\n".join(lines).strip(), build_system_message(
        prompt_mode,
        input_modality=input_modality,
        text_context_source=text_context_source,
    )
