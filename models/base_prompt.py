PROMPT_FORMATS = [
    'CQM-A',
    'CQM-LA',
    'CQM-EA',
    'CQM-LEA',
    'CQM-ELA',
    'CQM-AL',
    'CQM-AE',
    'CQM-ALE',
    'QCM-A',
    'QCM-LA',
    'QCM-EA',
    'QCM-LEA',
    'QCM-ELA',
    'QCM-AL',
    'QCM-AE',
    'QCM-ALE',
    'QCML-A',
    'QCME-A',
    'QCMLE-A',
    'QCLM-A',
    'QCEM-A',
    'QCLEM-A',
    'QCML-AE',
]


def get_question_text(problem):
    return problem['question']


def get_context_text(problem, use_caption=False, extra_context=None):
    parts = []
    if problem.get('hint'):
        parts.append(problem['hint'])
    if use_caption and problem.get('caption'):
        parts.append(problem['caption'])
    if extra_context:
        parts.append(extra_context)

    context = " ".join(parts).strip()
    return context or "N/A"


def get_choice_text(problem, options):
    choices = problem['choices']
    choice_list = []
    for i, choice in enumerate(choices):
        choice_list.append(f"({options[i]}) {choice}")
    return " ".join(choice_list)


def get_answer(problem, options):
    return options[problem['answer']]


def get_lecture_text(problem):
    return problem['lecture'].replace("\n", "\\n")


def get_solution_text(problem):
    return problem['solution'].replace("\n", "\\n")


def create_one_example(prompt_format,
                       question,
                       context,
                       choice,
                       answer,
                       lecture,
                       solution,
                       test_example=True):
    input_format, output_format = prompt_format.split("-")

    if input_format == "CQM":
        prompt_input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n"
    elif input_format == "QCM":
        prompt_input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
    elif input_format == "QCML":
        prompt_input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture}\n"
    elif input_format == "QCME":
        prompt_input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n"
    elif input_format == "QCMLE":
        prompt_input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture} {solution}\n"
    elif input_format == "QCLM":
        prompt_input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choice}\n"
    elif input_format == "QCEM":
        prompt_input = f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choice}\n"
    elif input_format == "QCLEM":
        prompt_input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choice}\n"
    else:
        raise ValueError(f"Unsupported prompt format: {prompt_format}")

    if test_example:
        prompt_output = "Answer:"
    elif output_format == 'A':
        prompt_output = f"Answer: The answer is {answer}."
    elif output_format == 'AL':
        prompt_output = f"Answer: The answer is {answer}. BECAUSE: {solution}"
    elif output_format == 'AE':
        prompt_output = f"Answer: The answer is {answer}. BECAUSE: {lecture}"
    elif output_format == 'ALE':
        prompt_output = f"Answer: The answer is {answer}. BECAUSE: {lecture} {solution}"
    elif output_format == 'AEL':
        prompt_output = f"Answer: The answer is {answer}. BECAUSE: {solution} {lecture}"
    elif output_format == 'LA':
        prompt_output = f"Answer: {lecture} The answer is {answer}."
    elif output_format == 'EA':
        prompt_output = f"Answer: {solution} The answer is {answer}."
    elif output_format == 'LEA':
        prompt_output = f"Answer: {lecture} {solution} The answer is {answer}."
    elif output_format == 'ELA':
        prompt_output = f"Answer: {solution} {lecture} The answer is {answer}."
    else:
        raise ValueError(f"Unsupported prompt format: {prompt_format}")

    text = (prompt_input + prompt_output).replace("  ", " ").strip()
    if text.endswith("BECAUSE:"):
        text = text[:-8].strip()
    return text


def build_prompt_examples(problems,
                          shot_qids,
                          test_qid,
                          args,
                          test_context_override=None,
                          shot_context_overrides=None):
    examples = []
    shot_context_overrides = shot_context_overrides or {}

    for qid in shot_qids:
        problem = problems[qid]
        context = shot_context_overrides.get(qid)
        if context is None:
            context = get_context_text(problem, use_caption=args.use_caption)

        train_example = create_one_example(
            args.prompt_format,
            get_question_text(problem),
            context,
            get_choice_text(problem, args.options),
            get_answer(problem, args.options),
            get_lecture_text(problem),
            get_solution_text(problem),
            test_example=False,
        )
        examples.append({
            "qid": qid,
            "text": train_example,
            "is_test": False,
        })

    test_problem = problems[test_qid]
    test_context = test_context_override
    if test_context is None:
        test_context = get_context_text(test_problem, use_caption=args.use_caption)

    test_example = create_one_example(
        args.prompt_format,
        get_question_text(test_problem),
        test_context,
        get_choice_text(test_problem, args.options),
        get_answer(test_problem, args.options),
        get_lecture_text(test_problem),
        get_solution_text(test_problem),
        test_example=True,
    )
    examples.append({
        "qid": test_qid,
        "text": test_example,
        "is_test": True,
    })
    return examples


def build_prompt(problems,
                 shot_qids,
                 test_qid,
                 args,
                 test_context_override=None,
                 shot_context_overrides=None):
    examples = build_prompt_examples(
        problems,
        shot_qids,
        test_qid,
        args,
        test_context_override=test_context_override,
        shot_context_overrides=shot_context_overrides,
    )
    return '\n\n'.join(example["text"] for example in examples)


def build_system_message(prompt_format, has_image=False):
    output_format = prompt_format.split("-")[1]
    parts = [
        "You solve ScienceQA multiple-choice questions.",
        "Follow the few-shot format exactly.",
        "Do not add markdown, preamble, or extra commentary.",
        "Use the available option letters only.",
    ]

    if output_format == "A":
        parts.append("For the final question, answer with the format: Answer: The answer is X.")
    else:
        parts.append("For the final question, start with 'Answer:' and keep the answer format consistent with the examples.")

    if has_image:
        parts.append("Each attached image appears immediately after the corresponding example or question text. Use each image only for its paired example or question.")

    return " ".join(parts)
