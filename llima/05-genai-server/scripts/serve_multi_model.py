#!/usr/bin/env python3
"""Serve multiple GenAI models (LLM + VLM + ASR) from one process.

This is the in-process ``pyneat.genai.GenAIServer`` (NOT the ``llima
benchmark-server`` CLI). It hosts several deployed model directories behind one
OpenAI-compatible HTTP endpoint. Clients pick a model with the ``model`` field
of each request; the served names registered here are ``llm``, ``vlm``, ``asr``.

Adapted from ``/workspace/core/tutorials/021_serve_genai_models/serve_genai_models.py``.
API surface verified against:
  - /workspace/core/include/genai/GenAIServer.h
  - /workspace/core/python/src/module.cpp  (GenAIServer bindings, ~line 2216)

Starting the server loads every registered model onto the MLA, so budget memory
before you run it. Launch it on the DevKit when you are ready:

  dk /workspace/demo-neat/llima/05-genai-server/scripts/serve_multi_model.py

Before serving fresh models, check the DevKit disk first with ``df -h /``. A GenAI
model directory can be several GB and the DevKit root filesystem is small, so
confirm free space before any ``llima pull``. The three defaults below are already
deployed, so this happy path needs no pull.
"""
import argparse
import time

import pyneat as neat

# Deployed model directories already on the board (see llima list).
# llima pull writes to /media/nvme/llima/models/<model-id>.
DEFAULT_LLM = "/media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4"
DEFAULT_VLM = "/media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4"
DEFAULT_ASR = "/media/nvme/llima/models/whisper-small-a16w8"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9998)
    parser.add_argument("--llm", default=DEFAULT_LLM,
                        help="LLM model directory (served name: llm). Empty string disables.")
    parser.add_argument("--vlm", default=DEFAULT_VLM,
                        help="VLM model directory (served name: vlm). Empty string disables.")
    parser.add_argument("--asr", default=DEFAULT_ASR,
                        help="ASR model directory (served name: asr). Empty string disables.")
    args = parser.parse_args()

    if not any([args.llm, args.vlm, args.asr]):
        raise RuntimeError("provide at least one of --llm, --vlm, or --asr")

    # Configure the server (GenAIServer.h: GenAIServerOptions {host, port}).
    options = neat.genai.GenAIServerOptions()
    options.host = args.host
    options.port = args.port
    server = neat.genai.GenAIServer(options)

    # Register model directories with an explicit served name. add_model loads
    # the model; every registered model is resident at the same time, so budget
    # memory before adding all three (see 02_multi_model_server.ipynb).
    if args.llm:
        server.add_model(args.llm, "llm")
    if args.vlm:
        server.add_model(args.vlm, "vlm")
    if args.asr:
        server.add_model(args.asr, "asr")

    print("registered models:", ", ".join(server.model_names()))
    print(f"serving on http://{options.host}:{options.port}")
    print(f"try: curl http://<devkit-host>:{options.port}/v1/models")

    # start() is non-blocking (app owns the process lifetime); serve() would
    # block instead. We keep the process alive and stop cleanly on Ctrl-C.
    server.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
