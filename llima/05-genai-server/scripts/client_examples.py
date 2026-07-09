#!/usr/bin/env python3
"""OpenAI-compatible client examples for a running GenAIServer.

Three request shapes against one server (see serve_multi_model.py):
  text        -> POST /v1/chat/completions          (served name: llm)
  image       -> POST /v1/chat/completions + image  (served name: vlm)
  transcribe  -> POST /v1/audio/transcriptions       (served name: asr)

Merged and adapted verbatim in shape from the tutorial 021 request clients:
  /workspace/core/tutorials/021_serve_genai_models/request_chat_completion_text.py
  /workspace/core/tutorials/021_serve_genai_models/request_chat_completion_image.py
  /workspace/core/tutorials/021_serve_genai_models/request_audio_transcription.py

These talk to a server that is ALREADY running; they do not load a model
themselves. They are still gated behind an explicit --run flag so importing or
py_compiling this file never sends a request. The owner runs, e.g.:

  python client_examples.py --run text  --server-ip 192.168.135.203 "Explain REST."
  python client_examples.py --run image --server-ip 192.168.135.203 scene.jpg "What is this?"
  python client_examples.py --run transcribe --server-ip 192.168.135.203 speech.wav
"""
import argparse
import base64
import json
import mimetypes
from pathlib import Path


def print_stream(response) -> None:
    """Consume an SSE stream and print text plus server TTFT / TPS.

    Handles both the chat-completions delta shape and the audio.transcription
    text shape used by the tutorial 021 clients.
    """
    ttft = None
    tps_samples = []
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        event = json.loads(payload)

        if event.get("object") == "audio.transcription.error":
            raise RuntimeError(event.get("error", "audio transcription failed"))
        if "error" in event:
            error = event["error"]
            if isinstance(error, dict):
                error = error.get("message", error)
            raise RuntimeError(error)

        ttft = event.get("ttft", ttft)
        if "tps" in event:
            tps_samples.append(float(event["tps"]))

        # chat/completions: choices[0].delta.content ; transcriptions: text
        text = event.get("text", "")
        if not text:
            choice = event.get("choices", [{}])[0]
            text = choice.get("delta", {}).get("content", "")
        if text:
            print(text, end="", flush=True)

    print()
    if ttft is not None:
        print(f"server ttft: {ttft:.4f}s")
    if tps_samples:
        avg = sum(tps_samples) / len(tps_samples)
        print(f"server tps: avg={avg:.2f} min={min(tps_samples):.2f} "
              f"max={max(tps_samples):.2f} tokens/s")


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def request_text(args) -> None:
    import requests
    url = f"http://{args.server_ip}:{args.server_port}/v1/chat/completions"
    prompt = " ".join(args.args) if args.args else "Give me three tips for a small REST API."
    payload = {
        "model": args.model or "llm",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    if args.max_tokens is not None:
        payload["max_tokens"] = args.max_tokens
    print(f"model: {payload['model']}")
    response = requests.post(url, json=payload, stream=True, timeout=120)
    response.raise_for_status()
    print_stream(response)


def request_image(args) -> None:
    import requests
    if not args.args:
        raise SystemExit("image mode needs: <image_path> [prompt words...]")
    image_path = Path(args.args[0])
    prompt = " ".join(args.args[1:]) if len(args.args) > 1 else "What is the main subject of this image?"
    url = f"http://{args.server_ip}:{args.server_port}/v1/chat/completions"
    payload = {
        "model": args.model or "vlm",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
            ],
        }],
        "stream": True,
    }
    if args.max_tokens is not None:
        payload["max_tokens"] = args.max_tokens
    print(f"model: {payload['model']}")
    response = requests.post(url, json=payload, stream=True, timeout=120)
    response.raise_for_status()
    print_stream(response)


def request_transcribe(args) -> None:
    import requests
    if not args.args:
        raise SystemExit("transcribe mode needs: <audio_file.wav>")
    audio_path = Path(args.args[0])
    url = f"http://{args.server_ip}:{args.server_port}/v1/audio/transcriptions"
    print(f"model: {args.model or 'asr'}")
    with audio_path.open("rb") as audio:
        response = requests.post(
            url,
            data={"model": args.model or "asr", "language": args.language, "stream": "true"},
            files={"file": (audio_path.name, audio, "audio/wav")},
            timeout=120,
            stream=True,
        )
    response.raise_for_status()
    print_stream(response)


MODES = {"text": request_text, "image": request_image, "transcribe": request_transcribe}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=MODES, dest="mode",
                        help="Actually send the request (omit to only print help).")
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=9998)
    parser.add_argument("--model", default=None, help="Served name override (default: llm/vlm/asr).")
    parser.add_argument("--language", default="en")
    parser.add_argument("--max-tokens", type=int, default=None, dest="max_tokens")
    parser.add_argument("args", nargs="*", help="Prompt words / image path / audio path.")
    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        print("\nNothing sent: pass --run {text,image,transcribe} to actually call the server.")
        return 0

    MODES[args.mode](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
