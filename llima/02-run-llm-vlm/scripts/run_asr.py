#!/usr/bin/env python3
"""Minimal CLI: transcribe audio with pyneat.genai.ASRModel (audio file -> text).

Loading and running a model is heavy and hardware-bound. Run this on the Modalix DevKit:

  dk /workspace/demo-neat/llima/02-run-llm-vlm/scripts/run_asr.py \
    --model /media/nvme/llima/models/whisper-small-a16w8 \
    --audio /workspace/demo-neat/llima/02-run-llm-vlm/assets/speech.wav

The direct ASRModel path (constructor takes the model directory, request carries audio_file/language)
is from /workspace/core/include/genai/ASRModel.h + GenAITypes.h and module.cpp. NOTE: the CLI
`llima run` needs `--stt_model_path <elf-dir>` for ASR (trap #2); the Python ASRModel does NOT - it
takes the model directory in its constructor.
"""
import argparse
from pathlib import Path

import pyneat as neat


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe audio via pyneat.genai.ASRModel.")
    parser.add_argument("--model", required=True,
                        help="Path to a deployed ASR model directory (e.g. "
                             "/media/nvme/llima/models/whisper-small-a16w8)")
    parser.add_argument("--audio", required=True, help="Path to a WAV file (16 kHz mono is safe).")
    parser.add_argument("--language", default="en", help="Spoken language (default: en).")
    parser.add_argument("--stream", action="store_true", help="Stream the transcript progressively.")
    args = parser.parse_args()

    if not Path(args.audio).is_file():
        raise SystemExit(f"audio file not found: {args.audio}")

    model = neat.genai.ASRModel(args.model)
    print(f"model_id={model.model_id()} accepts_audio={model.accepts_audio()}")

    request = neat.genai.GenerationRequest()
    request.audio_file = args.audio
    request.language = args.language

    if args.stream:
        print("transcription: ", end="", flush=True)
        for token in model.stream(request):
            print(token.text, end="", flush=True)
            if token.is_final:
                break
        print()
    else:
        result = model.run(request)
        print(f"transcription: {result.text}")
        m = result.metrics
        print(f"[tokens={m.generated_tokens} tok/s={m.tokens_per_second:.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
