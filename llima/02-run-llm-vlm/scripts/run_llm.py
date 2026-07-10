#!/usr/bin/env python3
"""Minimal CLI: run an LLM with pyneat.genai.GenAIModel (text).

Loading and running a model is heavy and hardware-bound. Run this on the Modalix DevKit:

  dk /workspace/demo-neat/llima/02-run-llm-vlm/scripts/run_llm.py \
    --model /media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4

Adapted from /workspace/core/tutorials/019_run_an_llm/run_an_llm.py; API names verified against
/workspace/core/python/src/module.cpp and /workspace/core/include/genai/.
"""
import argparse

import pyneat as neat


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an LLM (text) via pyneat.genai.GenAIModel.")
    parser.add_argument("--model", required=True,
                        help="Path to a deployed LLiMa model directory (e.g. "
                             "/media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4)")
    parser.add_argument("--prompt",
                        default="Give me three practical tips for designing a small REST API.",
                        help="User prompt.")
    parser.add_argument("--system-prompt", default=None, help="Optional system instruction.")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Cap on generated tokens.")
    parser.add_argument("--stream", action="store_true", help="Stream tokens instead of run().")
    args = parser.parse_args()

    # GenAIModel auto-detects the task from the model directory.
    model = neat.genai.GenAIModel(args.model)
    print(f"model_id={model.model_id()} task={model.task()} "
          f"accepts_text={model.accepts_text()}")

    request = neat.genai.GenerationRequest()
    request.prompt = args.prompt
    if args.system_prompt:
        request.system_prompt = args.system_prompt
    request.max_new_tokens = args.max_new_tokens

    if args.stream:
        print("assistant: ", end="", flush=True)
        for token in model.stream(request):
            print(token.text, end="", flush=True)
            if token.is_final:
                break
        print()
    else:
        result = model.run(request)
        print(f"assistant: {result.text}")
        m = result.metrics
        print(f"[tokens={m.generated_tokens} ttft_s={m.time_to_first_token_s:.3f} "
              f"tok/s={m.tokens_per_second:.2f} finish={result.finish_reason}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
