"""Serve Qwen2.5-VL feedback through the minimal chat-completions interface."""

import argparse
import base64
import io
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

from PIL import Image


def decode_message(payload):
    messages = payload.get("messages")
    if not isinstance(messages, list) or len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError("Expected one user message")
    content = messages[0].get("content")
    if not isinstance(content, list) or [item.get("type") for item in content] != ["text", "image_url", "image_url"]:
        raise ValueError("Expected text, before image and after image")
    prompt = content[0].get("text")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Missing task and outcome prompt")
    images = []
    for item in content[1:]:
        url = item.get("image_url", {}).get("url", "")
        prefix = "data:image/jpeg;base64,"
        if not isinstance(url, str) or not url.startswith(prefix):
            raise ValueError("Expected JPEG data URL")
        data = base64.b64decode(url[len(prefix):], validate=True)
        with Image.open(io.BytesIO(data)) as image:
            images.append(image.convert("RGB"))
    return prompt, images


class QwenFeedbackModel:
    def __init__(self, model_path, device="cuda"):
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device is unavailable for Qwen feedback")
        self.torch = torch
        self.process_vision_info = process_vision_info
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to(device)
        self.model.eval()

    def generate(self, prompt, images, max_tokens):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": images[0]},
            {"type": "image", "image": images[1]},
        ]}]
        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[prompt_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",
        ).to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        completion = generated[0, inputs.input_ids.shape[1]:]
        return self.processor.decode(completion, skip_special_tokens=True).strip()


def make_handler(model, served_model_name, api_key):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, data):
            output = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(output)))
            self.end_headers()
            self.wfile.write(output)

        def do_GET(self):
            if self.path == "/health":
                self.respond(200, {"status": "ready", "model": served_model_name})
            else:
                self.respond(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.respond(404, {"error": "not found"})
                return
            if api_key and self.headers.get("Authorization") != "Bearer " + api_key:
                self.respond(401, {"error": "unauthorized"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 8_000_000:
                    raise ValueError("Request body size must be within 8 MB")
                payload = json.loads(self.rfile.read(size))
                if payload.get("model") != served_model_name:
                    raise ValueError("Requested model does not match served model")
                prompt, images = decode_message(payload)
                max_tokens = min(max(int(payload.get("max_tokens", 512)), 1), 768)
                answer = model.generate(prompt, images, max_tokens)
            except (ValueError, KeyError, TypeError) as error:
                self.respond(400, {"error": str(error)})
                return
            self.respond(200, {"model": served_model_name, "choices": [{"message": {"content": answer}}]})

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    api_key = os.environ.get("QWEN_SERVER_API_KEY", "")
    if args.host not in {"127.0.0.1", "::1", "localhost"} and not api_key:
        raise ValueError("Non-loopback feedback service requires QWEN_SERVER_API_KEY")
    model = QwenFeedbackModel(args.model_path, args.device)
    handler = make_handler(model, args.served_model_name, api_key)
    print(f"Qwen feedback ready on {args.host}:{args.port} ({args.device})", flush=True)
    HTTPServer((args.host, args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
