import base64
import concurrent.futures
import http.client
import json
import mimetypes
import socket
import ssl
import urllib.error
import urllib.request


class APIClientError(RuntimeError):
    pass


def _read_text_blocks(blocks):
    if isinstance(blocks, str):
        return blocks.strip()

    if not isinstance(blocks, list):
        return ""

    texts = []
    for block in blocks:
        if isinstance(block, str):
            texts.append(block)
            continue

        if not isinstance(block, dict):
            continue

        text = block.get("text")
        if isinstance(text, str):
            texts.append(text)
            continue

        value = block.get("value")
        if isinstance(value, str):
            texts.append(value)

    return "".join(texts).strip()


def image_path_to_data_url(image_path):
    mime_type = mimetypes.guess_type(image_path)[0] or "application/octet-stream"
    with open(image_path, "rb") as file_obj:
        encoded = base64.b64encode(file_obj.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def merge_system_message(prompt_text, system_message):
    if not system_message:
        return prompt_text
    return f"Instruction:\n{system_message}\n\n{prompt_text}"


def normalize_content_blocks(prompt_text=None, image_path=None, content_blocks=None):
    if content_blocks is not None:
        return [dict(block) for block in content_blocks]

    blocks = [{"type": "text", "text": prompt_text}]
    if image_path:
        blocks.append({"type": "image_path", "image_path": image_path})
    return blocks


def inject_system_message_into_content_blocks(content_blocks, system_message):
    if not system_message:
        return content_blocks

    blocks = [dict(block) for block in content_blocks]
    if blocks and blocks[0].get("type") == "text":
        blocks[0]["text"] = merge_system_message(blocks[0]["text"], system_message)
    else:
        blocks.insert(0, {"type": "text", "text": f"Instruction:\n{system_message}"})
    return blocks


class InferenceAPIClient:

    def __init__(self,
                 base_url,
                 endpoint_type,
                 model,
                 api_key,
                 timeout=120,
                 temperature=0.0,
                 top_p=1.0,
                 max_output_tokens=512,
                 stream=False,
                 store=False,
                 response_format=None,
                 reasoning_effort=None,
                 image_detail="auto",
                 user_agent="ScienceQA-Inference/1.0"):
        self.base_url = base_url.rstrip("/")
        self.endpoint_type = endpoint_type
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.stream = stream
        self.store = store
        self.response_format = response_format
        self.reasoning_effort = reasoning_effort
        self.image_detail = image_detail
        self.user_agent = user_agent

    def build_payload(self, prompt_text=None, system_message=None, image_path=None, content_blocks=None):
        content_blocks = normalize_content_blocks(
            prompt_text=prompt_text,
            image_path=image_path,
            content_blocks=content_blocks,
        )
        if self.endpoint_type == "responses":
            content_blocks = inject_system_message_into_content_blocks(content_blocks, system_message)
            return self._build_responses_payload(content_blocks)
        if self.endpoint_type == "chat_completions":
            return self._build_chat_payload(content_blocks, system_message)

        raise ValueError(f"Unsupported endpoint type: {self.endpoint_type}")

    def generate(self, prompt_text=None, system_message=None, image_path=None, content_blocks=None):
        payload = self.build_payload(
            prompt_text=prompt_text,
            system_message=system_message,
            image_path=image_path,
            content_blocks=content_blocks,
        )
        if self.stream:
            return self._stream_request(payload)
        return self._json_request(payload)

    def list_models(self):
        response_data = self._get_json(f"{self.base_url}/models")
        models = response_data.get("data")
        if not isinstance(models, list):
            raise APIClientError(f"Unable to extract models from response: {json.dumps(response_data)[:1000]}")

        model_ids = []
        for item in models:
            if not isinstance(item, dict):
                continue
            model_id = item.get("id")
            if isinstance(model_id, str) and model_id:
                model_ids.append(model_id)

        if not model_ids:
            raise APIClientError(f"Model list is empty or invalid: {json.dumps(response_data)[:1000]}")
        return model_ids

    def _build_responses_payload(self, content_blocks):
        content = self._serialize_responses_content(content_blocks)

        items = [{
            "type": "message",
            "role": "user",
            "content": content,
        }]

        payload = {
            "model": self.model,
            "input": items,
            "store": self.store,
            "stream": self.stream,
        }
        if self.max_output_tokens:
            payload["max_output_tokens"] = self.max_output_tokens
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.reasoning_effort is not None:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        return payload

    def _build_chat_payload(self, content_blocks, system_message):
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})

        user_content = self._serialize_chat_content(content_blocks)
        if len(user_content) == 1 and user_content[0]["type"] == "text":
            messages.append({"role": "user", "content": user_content[0]["text"]})
        else:
            messages.append({"role": "user", "content": user_content})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": self.stream,
        }
        if self.max_output_tokens:
            payload["max_tokens"] = self.max_output_tokens
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.response_format is not None:
            payload["response_format"] = self.response_format
        if self.reasoning_effort is not None:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        return payload

    def _serialize_responses_content(self, content_blocks):
        content = []
        for block in content_blocks:
            if block["type"] == "text":
                content.append({"type": "input_text", "text": block["text"]})
            elif block["type"] == "image_path":
                content.append({
                    "type": "input_image",
                    "image_url": image_path_to_data_url(block["image_path"]),
                    "detail": self.image_detail,
                })
            else:
                raise ValueError(f"Unsupported content block type: {block['type']}")
        return content

    def _serialize_chat_content(self, content_blocks):
        content = []
        for block in content_blocks:
            if block["type"] == "text":
                content.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image_path":
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_path_to_data_url(block["image_path"]),
                        "detail": self.image_detail,
                    },
                })
            else:
                raise ValueError(f"Unsupported content block type: {block['type']}")
        return content

    def _json_request(self, payload):
        response_data = self._request(payload)
        return self._extract_text(response_data)

    def _stream_request_inner(self, payload):
        endpoint = self._get_endpoint_url()
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=data,
            headers=self._build_headers(),
            method="POST",
        )

        deltas = []
        final_text = ""
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data:"):
                        continue

                    body = line[5:].strip()
                    if body == "[DONE]":
                        break

                    event = json.loads(body)
                    delta = self._extract_stream_delta(event)
                    if delta:
                        deltas.append(delta)

                    candidate = self._extract_stream_final_text(event)
                    if candidate:
                        final_text = candidate
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise APIClientError(f"HTTP {exc.code}: {message}") from exc
        except urllib.error.URLError as exc:
            raise APIClientError(str(exc)) from exc
        except http.client.RemoteDisconnected as exc:
            raise APIClientError(str(exc)) from exc
        except ConnectionResetError as exc:
            raise APIClientError(str(exc)) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise APIClientError(str(exc)) from exc
        except ssl.SSLError as exc:
            raise APIClientError(f"SSL error: {exc}") from exc

        accumulated = "".join(deltas).strip()
        final = final_text.strip()
        if final and accumulated and final not in accumulated and not accumulated.endswith(final):
            text = accumulated + "\n" + final
        else:
            text = accumulated or final
        if not text:
            raise APIClientError("No text was returned from the streaming response.")
        return text

    def _stream_request(self, payload):
        return self._stream_request_inner(payload)

    def _request_inner(self, payload):
        endpoint = self._get_endpoint_url()
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=data,
            headers=self._build_headers(),
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise APIClientError(f"HTTP {exc.code}: {message}") from exc
        except urllib.error.URLError as exc:
            raise APIClientError(str(exc)) from exc
        except http.client.RemoteDisconnected as exc:
            raise APIClientError(str(exc)) from exc
        except ConnectionResetError as exc:
            raise APIClientError(str(exc)) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise APIClientError(str(exc)) from exc
        except ssl.SSLError as exc:
            raise APIClientError(f"SSL error: {exc}") from exc

    def _request(self, payload):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._request_inner, payload)
            try:
                return future.result(timeout=self.timeout)
            except concurrent.futures.TimeoutError:
                raise APIClientError(f"Request timed out after {self.timeout}s (hard deadline)")

    def _get_json(self, endpoint):
        request = urllib.request.Request(
            endpoint,
            headers=self._build_headers(),
            method="GET",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise APIClientError(f"HTTP {exc.code}: {message}") from exc
        except urllib.error.URLError as exc:
            raise APIClientError(str(exc)) from exc
        except http.client.RemoteDisconnected as exc:
            raise APIClientError(str(exc)) from exc
        except ConnectionResetError as exc:
            raise APIClientError(str(exc)) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise APIClientError(str(exc)) from exc
        except ssl.SSLError as exc:
            raise APIClientError(f"SSL error: {exc}") from exc

    def _extract_text(self, response_data):
        if self.endpoint_type == "responses":
            text = self._extract_responses_text(response_data)
        else:
            text = self._extract_chat_text(response_data)

        if not text:
            raise APIClientError(f"Unable to extract text from response: {json.dumps(response_data)[:1000]}")
        return text

    def _extract_responses_text(self, response_data):
        output_text = response_data.get("output_text")
        if output_text:
            text = _read_text_blocks(output_text)
            if text:
                return text

        texts = []
        for item in response_data.get("output", []):
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                if content.get("type") in ("output_text", "text"):
                    text = content.get("text") or content.get("value")
                    if isinstance(text, str):
                        texts.append(text)

        return "".join(texts).strip()

    def _extract_chat_text(self, response_data):
        choices = response_data.get("choices") or []
        if not choices:
            return ""

        message = choices[0].get("message", {})
        content_text = _read_text_blocks(message.get("content", ""))
        if content_text:
            return content_text
        return _read_text_blocks(message.get("reasoning", ""))

    def _extract_stream_delta(self, event):
        if self.endpoint_type == "responses":
            if event.get("type") == "response.output_text.delta":
                return event.get("delta", "")
            return ""

        choices = event.get("choices") or []
        if not choices:
            return ""

        delta_obj = choices[0].get("delta", {})
        text = _read_text_blocks(delta_obj.get("content", ""))
        if not text:
            text = _read_text_blocks(delta_obj.get("reasoning_content", ""))
        return text

    def _extract_stream_final_text(self, event):
        if self.endpoint_type == "responses":
            if event.get("type") == "response.output_text.done":
                return event.get("text", "")
            if event.get("type") == "response.completed":
                response = event.get("response", {})
                return self._extract_responses_text(response)
            return ""

        choices = event.get("choices") or []
        if not choices:
            return ""

        message = choices[0].get("message")
        if not message:
            return ""
        text = _read_text_blocks(message.get("content", ""))
        if not text:
            text = _read_text_blocks(message.get("reasoning_content", ""))
        return text

    def _build_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }

    def _get_endpoint_url(self):
        if self.endpoint_type == "responses":
            return f"{self.base_url}/responses"
        return f"{self.base_url}/chat/completions"
