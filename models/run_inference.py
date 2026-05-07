import argparse
import json
import os
import random
import re
import sys
import time


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.base_prompt import PROMPT_FORMATS, build_prompt, build_prompt_examples, build_system_message
from models.inference_api import APIClientError, InferenceAPIClient


IMAGE_INPUT_MODES = {"image", "caption_image"}


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def resolve_input_mode(args):
    if args.input_mode:
        return args.input_mode
    return "caption" if args.use_caption else "text"


def load_data(args):
    problems = load_json(os.path.join(args.data_root, "problems.json"))
    pid_splits = load_json(os.path.join(args.data_root, "pid_splits.json"))

    captions = {}
    if os.path.exists(args.caption_file):
        captions = load_json(args.caption_file).get("captions", {})
    elif resolve_input_mode(args) in {"caption", "caption_image"}:
        raise FileNotFoundError(f"Caption file not found: {args.caption_file}")

    for qid, problem in problems.items():
        problem["caption"] = captions.get(qid, "")

    qids = pid_splits[args.test_split]
    if resolve_input_mode(args) in IMAGE_INPUT_MODES:
        qids = [qid for qid in qids if problems[qid].get("image")]
    if args.test_number > 0:
        qids = qids[:args.test_number]
    print(f"number of test problems: {len(qids)}\n")

    shot_qids = args.shot_qids
    train_qids = pid_splits["train"]
    if resolve_input_mode(args) in IMAGE_INPUT_MODES:
        train_qids = [qid for qid in train_qids if problems[qid].get("image")]
    if shot_qids is None:
        if not 0 <= args.shot_number <= 32:
            raise ValueError("shot_number must be between 0 and 32.")
        if len(train_qids) < args.shot_number:
            raise ValueError("Not enough image-based training examples for the requested shot_number.")
        shot_qids = random.sample(train_qids, args.shot_number)
    else:
        shot_qids = [str(qid) for qid in shot_qids]
        for qid in shot_qids:
            if qid not in train_qids:
                raise ValueError(f"Shot question id is not in train split: {qid}")

    print("training question ids for prompting: ", shot_qids, "\n")
    return problems, qids, shot_qids


def resolve_api_key_info(args, allow_dummy=False):
    if args.api_key:
        return args.api_key, "cli"

    candidate_names = [args.api_key_env, "INFINITEAI_API_KEY", "OPENAI_API_KEY"]
    for name in candidate_names:
        if not name:
            continue
        value = os.getenv(name)
        if value:
            return value, f"env:{name}"

    if allow_dummy:
        return "DUMMY_KEY", "dummy"

    raise ValueError("API key not found. Use --api_key or set the configured environment variable.")


def sanitize_args_for_logging(args):
    payload = vars(args).copy()
    if payload.get("api_key"):
        payload["api_key"] = "***"
    return payload


def sanitize_path_component(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def get_result_file(args):
    model_dir = args.result_dir or sanitize_path_component(args.model)
    file_name = "{}_{}_{}_{}_test_{}_seed_{}.json".format(
        args.label,
        args.test_split,
        args.prompt_format,
        args.shot_number,
        args.test_number,
        args.seed,
    )
    return os.path.join(args.output_root, model_dir, file_name)


def collect_wrong_qids(results, problems, qids):
    wrong_qids = []
    for qid in qids:
        if qid not in results:
            continue
        if results[qid] != problems[qid]["answer"]:
            wrong_qids.append(qid)
    return wrong_qids


def save_results(result_file, acc, correct, count, shot_qids, args, results, outputs, wrong_qids):
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    safe_args = sanitize_args_for_logging(args)
    payload = {
        "acc": acc,
        "correct": correct,
        "count": count,
        "shot_qids": shot_qids,
        "args": safe_args,
        "results": results,
        "outputs": outputs,
        "wrong_qids": wrong_qids,
    }
    with open(result_file, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, separators=(",", ": "), ensure_ascii=False)


def redact_payload_for_preview(payload):
    if isinstance(payload, dict):
        redacted = {}
        for key, value in payload.items():
            if key == "image_url" and isinstance(value, str) and value.startswith("data:"):
                redacted[key] = "<data-url omitted>"
            elif key == "image_url" and isinstance(value, dict):
                nested = dict(value)
                if isinstance(nested.get("url"), str) and nested["url"].startswith("data:"):
                    nested["url"] = "<data-url omitted>"
                redacted[key] = redact_payload_for_preview(nested)
            else:
                redacted[key] = redact_payload_for_preview(value)
        return redacted

    if isinstance(payload, list):
        return [redact_payload_for_preview(item) for item in payload]

    return payload


def print_dry_run_request(prompt, system_message, image_path, content_blocks, payload, endpoint_type):
    print("====Dry Run Request Summary====")
    if content_blocks:
        if system_message:
            if endpoint_type == "responses":
                print("[instruction -> injected into first text block]")
            else:
                print("[system]")
            print(system_message)
            print("")
        for index, block in enumerate(content_blocks, 1):
            print(f"[block {index}] type={block['type']}")
            if block["type"] == "text":
                print(block["text"])
            elif block["type"] == "image_path":
                print(block["image_path"])
            print("")
    else:
        if system_message:
            print("[system]")
            print(system_message)
            print("")
        print("[prompt]")
        print(prompt)
        print("")
        if image_path:
            print("[image_path]")
            print(image_path)
            print("")

    print("====Dry Run Payload (redacted)====")
    print(json.dumps(redact_payload_for_preview(payload), indent=2, ensure_ascii=False))


def extract_prediction(text, valid_options):
    upper_text = text.upper()
    normalized_text = upper_text
    normalized_text = normalized_text.replace("**", "")
    normalized_text = normalized_text.replace("__", "")
    normalized_text = normalized_text.replace("`", "")
    normalized_text = normalized_text.replace("“", "\"").replace("”", "\"")
    normalized_text = normalized_text.replace("（", "(").replace("）", ")")
    normalized_text = normalized_text.replace("：", ":")

    patterns = [
        r"THE CORRECT ANSWER IS\s*\(?([A-Z])\)?",
        r"THE ANSWER IS\s*\(?([A-Z])\)?",
        r"CORRECT ANSWER\s*:\s*\(?([A-Z])\)?",
        r"ANSWER\s*:\s*THE ANSWER IS\s*\(?([A-Z])\)?",
        r"ANSWER\s*:\s*\(?([A-Z])\)?",
        r"OPTION\s*\(?([A-Z])\)?",
        r"CHOICE\s*\(?([A-Z])\)?",
        r"^\s*\(?([A-Z])\)?(?:[\s\.:]|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized_text, flags=re.MULTILINE)
        if match:
            candidate = match.group(1)
            if candidate in valid_options:
                return candidate

    for raw_line in normalized_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        line = re.sub(r"^[>\-#\*\s]+", "", line)
        line = re.sub(r"^\[(?:ANSWER|OPTION|CHOICE)\]\s*", "", line)

        leading_match = re.match(r"^\(?([A-Z])\)?(?=[\s\)\].:\-]|$)", line)
        if leading_match:
            candidate = leading_match.group(1)
            if candidate in valid_options:
                return candidate

    return "FAILED"


def get_pred_idx(prediction, choices, options):
    valid_options = options[:len(choices)]
    if prediction in valid_options:
        return options.index(prediction)
    return random.choice(range(len(choices)))


def resolve_image_path(problem, qid, args):
    if resolve_input_mode(args) not in IMAGE_INPUT_MODES:
        return None
    image_name = problem.get("image")
    if not image_name:
        return None

    image_path = os.path.join(args.image_root, problem["split"], qid, image_name)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Expected image file was not found: {image_path}")
    return image_path


def build_test_context_override(problem, image_path, args):
    input_mode = resolve_input_mode(args)
    use_caption = input_mode in {"caption", "caption_image"}
    context_parts = []
    if problem.get("hint"):
        context_parts.append(problem["hint"])
    if use_caption and problem.get("caption"):
        context_parts.append(problem["caption"])
    if image_path:
        context_parts.append("Use the attached image paired with this question.")
    context = " ".join(context_parts).strip()
    return context or "N/A"


def create_client(args):
    api_key, api_key_source = resolve_api_key_info(args, allow_dummy=args.dry_run)
    args.api_key_source = api_key_source
    return InferenceAPIClient(
        base_url=args.api_base_url,
        endpoint_type=args.endpoint_type,
        model=args.model,
        api_key=api_key,
        timeout=args.timeout,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_tokens,
        stream=args.stream,
        store=args.store,
        image_detail=args.image_detail,
        user_agent=args.user_agent,
    )


def build_multimodal_content_blocks(examples, image_paths_by_qid):
    blocks = []
    for example in examples:
        blocks.append({"type": "text", "text": example["text"]})
        image_path = image_paths_by_qid.get(example["qid"])
        if image_path:
            blocks.append({"type": "image_path", "image_path": image_path})
    return blocks


def build_problem_request(problems, shot_qids, qid, args):
    problem = problems[qid]
    input_mode = resolve_input_mode(args)
    uses_direct_image = input_mode in IMAGE_INPUT_MODES

    image_paths_by_qid = {}
    shot_context_overrides = {}
    if uses_direct_image:
        for shot_qid in shot_qids:
            shot_problem = problems[shot_qid]
            shot_image_path = resolve_image_path(shot_problem, shot_qid, args)
            image_paths_by_qid[shot_qid] = shot_image_path
            shot_context_overrides[shot_qid] = build_test_context_override(shot_problem, shot_image_path, args)

    image_path = resolve_image_path(problem, qid, args)
    if uses_direct_image:
        image_paths_by_qid[qid] = image_path

    test_context_override = build_test_context_override(problem, image_path, args)
    examples = build_prompt_examples(
        problems,
        shot_qids,
        qid,
        args,
        test_context_override=test_context_override,
        shot_context_overrides=shot_context_overrides,
    )

    prompt = '\n\n'.join(example["text"] for example in examples)
    content_blocks = None
    if uses_direct_image:
        content_blocks = build_multimodal_content_blocks(examples, image_paths_by_qid)

    system_message = build_system_message(args.prompt_format, has_image=uses_direct_image)
    if args.system_message:
        system_message = f"{system_message} {args.system_message}".strip()
    return prompt, system_message, image_path, content_blocks


def run_inference(client, prompt, system_message, image_path, content_blocks, args):
    last_error = None
    for attempt in range(args.max_retries + 1):
        try:
            return client.generate(
                prompt_text=prompt,
                system_message=system_message,
                image_path=image_path,
                content_blocks=content_blocks,
            )
        except APIClientError as exc:
            last_error = exc
            if attempt >= args.max_retries:
                break
            wait_seconds = max(args.retry_sleep, 0)
            if wait_seconds:
                time.sleep(wait_seconds)
    raise last_error


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=os.path.join(PROJECT_ROOT, "data", "scienceqa"))
    parser.add_argument("--image_root", type=str, default=os.path.join(PROJECT_ROOT, "data", "scienceqa", "images"))
    parser.add_argument("--output_root", type=str, default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument("--caption_file", type=str, default=os.path.join(PROJECT_ROOT, "data", "captions.json"))
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--options", nargs="+", default=["A", "B", "C", "D", "E"])

    parser.add_argument("--label", type=str, default="exp0")
    parser.add_argument("--test_split", type=str, default="val", choices=["test", "val", "minival", "minitest"])
    parser.add_argument("--test_number", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--prompt_format", type=str, default="CQM-A", choices=PROMPT_FORMATS)
    parser.add_argument("--shot_number", type=int, default=0)
    parser.add_argument("--shot_qids", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=10)

    parser.add_argument("--use_caption", action="store_true", help="Backward-compatible alias for caption input mode.")
    parser.add_argument("--input_mode",
                        type=str,
                        default=None,
                        choices=["text", "caption", "image", "caption_image"],
                        help="Choose text-only, caption text, direct image, or caption plus direct image.")

    parser.add_argument("--api_base_url", type=str, default="<API_BASE_URL_PLACEHOLDER>")
    parser.add_argument("--endpoint_type", type=str, default="responses", choices=["responses", "chat_completions"])
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_key_env", type=str, default="INFINITEAI_API_KEY")
    parser.add_argument("--system_message", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=4.0)
    parser.add_argument("--image_detail", type=str, default="auto")
    parser.add_argument("--user_agent", type=str, default="ScienceQA-Inference/1.0")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--store", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Print the request payload for the first example and exit.")

    args = parser.parse_args()
    if args.input_mode is None:
        args.input_mode = resolve_input_mode(args)
    args.use_caption = args.input_mode in {"caption", "caption_image"}
    return args


def main():
    args = parse_args()
    _, api_key_source = resolve_api_key_info(args, allow_dummy=args.dry_run)
    args.api_key_source = api_key_source
    print("====Input Arguments====")
    print(json.dumps(sanitize_args_for_logging(args), indent=2, ensure_ascii=False))

    random.seed(args.seed)

    problems, qids, shot_qids = load_data(args)
    client = create_client(args)
    result_file = get_result_file(args)

    if os.path.exists(result_file):
        print("# The result file exists! We will load the check point!!!")
        check_point = load_json(result_file)
        correct = check_point["correct"]
        results = check_point["results"]
        outputs = check_point["outputs"]
        print(f"{len(results)}/{len(qids)}, correct: {correct}, acc: {round(check_point['acc'], 2)}%")
    else:
        correct = 0
        results = {}
        outputs = {}

    if args.dry_run:
        sample_qid = qids[0]
        if args.input_mode in IMAGE_INPUT_MODES:
            for candidate_qid in qids:
                if problems[candidate_qid].get("image"):
                    sample_qid = candidate_qid
                    break
        prompt, system_message, image_path, content_blocks = build_problem_request(problems, shot_qids, sample_qid, args)
        payload = client.build_payload(
            prompt_text=prompt,
            system_message=system_message,
            image_path=image_path,
            content_blocks=content_blocks,
        )
        print_dry_run_request(prompt, system_message, image_path, content_blocks, payload, args.endpoint_type)
        return

    for index, qid in enumerate(qids):
        if qid in results:
            continue

        problem = problems[qid]
        choices = problem["choices"]
        answer = problem["answer"]
        label = args.options[answer]

        prompt, system_message, image_path, content_blocks = build_problem_request(problems, shot_qids, qid, args)
        output = run_inference(client, prompt, system_message, image_path, content_blocks, args)
        prediction = extract_prediction(output, args.options[:len(choices)])
        pred_idx = get_pred_idx(prediction, choices, args.options)

        results[qid] = pred_idx
        outputs[qid] = output
        if pred_idx == answer:
            correct += 1

        acc = correct / len(results) * 100

        if args.debug or index < 3:
            print("##################################")
            print(prompt, "\n")
            if image_path:
                print("# image path:", image_path)
            print("# labeled answer:", label)
            print("# predicted answer:", prediction)
            print("# predicted index:", pred_idx)
            print("# predicted output:", output)

        if (index + 1) % args.save_every == 0 or (index + 1) == len(qids):
            print(f"{len(results)}/{len(qids)}, correct: {correct}, acc: {round(acc, 2)}%, saving to {result_file}")
            wrong_qids = collect_wrong_qids(results, problems, qids)
            save_results(result_file, acc, correct, len(results), shot_qids, args, results, outputs, wrong_qids)


if __name__ == "__main__":
    main()
