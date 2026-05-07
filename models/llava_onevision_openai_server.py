from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class RequestValidationError(ValueError):
    pass


def decode_data_url(data_url: str) -> dict[str, Any]:
    if not isinstance(data_url, str) or not data_url.startswith("data:"):
        raise RequestValidationError("Only data URL images are supported.")

    header, separator, encoded = data_url.partition(",")
    if separator != ",":
        raise RequestValidationError("Invalid data URL payload.")
    if ";base64" not in header:
        raise RequestValidationError("Only base64-encoded data URLs are supported.")

    mime_type = header[5:].split(";", 1)[0] or "application/octet-stream"
    try:
        raw_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RequestValidationError("Invalid base64 image payload.") from exc

    return {"mime_type": mime_type, "data": raw_bytes}


def _read_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if not isinstance(content, list):
        return ""

    texts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return "\n\n".join(texts)


def parse_chat_messages(messages: Any) -> dict[str, Any]:
    if not isinstance(messages, list) or not messages:
        raise RequestValidationError("`messages` must be a non-empty list.")

    system_parts: list[str] = []
    user_parts: list[str] = []
    images: list[dict[str, Any]] = []

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = message.get("content")
        if role == "system":
            system_text = _read_text_content(content)
            if system_text:
                system_parts.append(system_text)
            continue

        if role != "user":
            continue

        if isinstance(content, str):
            if content.strip():
                user_parts.append(content.strip())
            continue

        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    user_parts.append(text.strip())
            elif block_type == "image_url":
                image_value = block.get("image_url")
                image_url = image_value.get("url") if isinstance(image_value, dict) else image_value
                images.append(decode_data_url(image_url))

    prompt_text = "\n\n".join(user_parts).strip()
    if not prompt_text:
        raise RequestValidationError("No user text content was found in the chat payload.")

    return {
        "system_message": "\n\n".join(system_parts).strip(),
        "prompt_text": prompt_text,
        "images": images,
    }


def build_chat_completion_response(model_name: str, content: str, response_id: str | None = None) -> dict[str, Any]:
    response_id = response_id or f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
    }


def build_models_response(model_name: str) -> dict[str, Any]:
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": created,
                "owned_by": "local",
            }
        ],
    }


def _load_pil_image(image_item: dict[str, Any]) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to serve multimodal requests.") from exc

    image = Image.open(io.BytesIO(image_item["data"]))
    return image.convert("RGB")


class LlavaOnevisionModelAdapter:
    def __init__(
        self,
        model_path: str,
        served_model_name: str,
        max_concurrent_requests: int = 1,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_path = model_path
        self.served_model_name = served_model_name
        self.max_concurrent_requests = max(1, int(max_concurrent_requests))
        self.trust_remote_code = trust_remote_code
        self._processor = None
        self._model = None
        self._load_lock = threading.Lock()
        self._generate_gate = threading.Semaphore(self.max_concurrent_requests)

    def list_models_payload(self) -> dict[str, Any]:
        return build_models_response(self.served_model_name)

    def generate_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("stream"):
            raise RequestValidationError("Streaming is not supported by this adapter server.")

        parsed = parse_chat_messages(payload.get("messages"))
        max_tokens = int(payload.get("max_tokens") or 512)
        temperature = payload.get("temperature", 0.0)
        top_p = payload.get("top_p", 1.0)

        with self._generate_gate:
            output_text = self.generate_text(
                prompt_text=parsed["prompt_text"],
                system_message=parsed["system_message"],
                images=parsed["images"],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        return build_chat_completion_response(
            model_name=self.served_model_name,
            content=output_text,
        )

    def generate_text(
        self,
        prompt_text: str,
        system_message: str,
        images: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        processor, model = self._ensure_loaded()

        merged_prompt = prompt_text
        if system_message:
            merged_prompt = f"Instruction:\n{system_message}\n\n{prompt_text}"

        content: list[dict[str, Any]] = []
        for image_item in images:
            content.append({"type": "image", "image": _load_pil_image(image_item)})
        content.append({"type": "text", "text": merged_prompt})

        messages = [{"role": "user", "content": content}]
        rendered_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        processor_kwargs: dict[str, Any] = {
            "text": [rendered_text],
            "return_tensors": "pt",
            "padding": True,
        }
        image_inputs, video_inputs = self._extract_vision_inputs(messages)
        if image_inputs is not None:
            processor_kwargs["images"] = image_inputs
        if video_inputs is not None:
            processor_kwargs["videos"] = video_inputs

        model_inputs = processor(**processor_kwargs)
        model_inputs = self._move_to_model_device(model_inputs, model)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
        }
        if float(temperature) > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = float(temperature)
            generation_kwargs["top_p"] = float(top_p)
        else:
            generation_kwargs["do_sample"] = False

        generated_ids = model.generate(**model_inputs, **generation_kwargs)
        input_ids = model_inputs["input_ids"]
        trimmed_ids = generated_ids[:, input_ids.shape[1]:]
        outputs = processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return outputs[0].strip()

    @staticmethod
    def _extract_vision_inputs(messages: list[dict[str, Any]]) -> tuple[Any, Any]:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            image_inputs = []
            for message in messages:
                for block in message.get("content", []):
                    if block.get("type") == "image":
                        image_inputs.append(block.get("image"))
            return (image_inputs or None), None

        return process_vision_info(messages)

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model

        with self._load_lock:
            if self._processor is not None and self._model is not None:
                return self._processor, self._model

            try:
                from transformers import AutoModelForCausalLM, AutoProcessor
            except ImportError as exc:
                raise RuntimeError("transformers is required to start the adapter server.") from exc

            self._processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=self.trust_remote_code,
            )
        return self._processor, self._model

    @staticmethod
    def _move_to_model_device(model_inputs: Any, model: Any) -> Any:
        if not hasattr(model_inputs, "to"):
            return model_inputs

        device = getattr(model, "device", None)
        if device is None:
            try:
                first_param = next(model.parameters())
            except (AttributeError, StopIteration, TypeError):
                return model_inputs
            device = first_param.device
        return model_inputs.to(device)


class OpenAICompatibleHandler(BaseHTTPRequestHandler):
    adapter: LlavaOnevisionModelAdapter | None = None

    def do_GET(self) -> None:
        if self.path == "/health":
            self._write_json(HTTPStatus.OK, {"status": "ok"})
            return
        if self.path == "/v1/models":
            self._write_json(HTTPStatus.OK, self._require_adapter().list_models_payload())
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": {"message": "Not found"}})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": {"message": "Not found"}})
            return

        try:
            payload = self._read_json_body()
            response = self._require_adapter().generate_from_payload(payload)
        except RequestValidationError as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            return
        except RuntimeError as exc:
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": {"message": str(exc), "type": "server_error"}})
            return
        except Exception as exc:  # pragma: no cover - safety net around model runtime
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": {"message": str(exc), "type": "server_error"}})
            return

        self._write_json(HTTPStatus.OK, response)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RequestValidationError("Request body must be valid JSON.") from exc

        if not isinstance(payload, dict):
            raise RequestValidationError("Top-level request payload must be a JSON object.")
        return payload

    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        response_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    @classmethod
    def _require_adapter(cls) -> LlavaOnevisionModelAdapter:
        if cls.adapter is None:
            raise RuntimeError("Model adapter has not been initialized.")
        return cls.adapter


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve LLaVA-OneVision-1.5 with an OpenAI-compatible chat endpoint.")
    parser.add_argument("--model-path", required=True, help="Local path of the LLaVA-OneVision-1.5 model.")
    parser.add_argument("--served-model-name", default="LLaVA-OneVision-1.5-4B-Instruct")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-concurrent-requests", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    return parser


def run_server(args: argparse.Namespace) -> None:
    OpenAICompatibleHandler.adapter = LlavaOnevisionModelAdapter(
        model_path=args.model_path,
        served_model_name=args.served_model_name,
        max_concurrent_requests=args.max_concurrent_requests,
        trust_remote_code=args.trust_remote_code,
    )
    with ThreadingHTTPServer((args.host, args.port), OpenAICompatibleHandler) as server:
        print(
            json.dumps(
                {
                    "status": "listening",
                    "host": args.host,
                    "port": args.port,
                    "model_path": args.model_path,
                    "served_model_name": args.served_model_name,
                    "max_concurrent_requests": args.max_concurrent_requests,
                },
                ensure_ascii=False,
            )
        )
        server.serve_forever()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_server(args)


if __name__ == "__main__":
    main()
