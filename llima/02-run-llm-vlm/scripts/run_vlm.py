#!/usr/bin/env python3
"""Minimal CLI: run a VLM with pyneat.genai.VisionLanguageModel (image + prompt -> text).

MANUAL RUN - this loads and runs a model. It is NOT executed by the training tooling.
Run it yourself on the Modalix DevKit.

  # Human, real terminal:
  dk /workspace/demo-neat/llima/02-run-llm-vlm/scripts/run_vlm.py \
    --model /media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4 \
    --image /workspace/demo-neat/tutorial/assets/images/image.png \
    --prompt "Describe this image in one sentence."

  # Automation (ssh + timeout):
  timeout 300 ssh -o BatchMode=yes sima@192.168.135.203 \
    'source $HOME/pyneat/bin/activate; python /workspace/demo-neat/llima/02-run-llm-vlm/scripts/run_vlm.py \
       --model /media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4 \
       --image /workspace/demo-neat/tutorial/assets/images/image.png'

Adapted from /workspace/core/tutorials/020_run_a_vlm/run_a_vlm.py; VLM images must be uint8 HWC RGB
(GenAITypes.h). OpenCV reads BGR, so we convert to RGB before building the request.
"""
import argparse

import cv2
import numpy as np
import pyneat as neat


def load_rgb_image(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read image: {path}")
    return np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))  # HWC RGB uint8


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a VLM via pyneat.genai.VisionLanguageModel.")
    parser.add_argument("--model", required=True, help="Path to a deployed VLM model directory.")
    parser.add_argument("--image", required=True, help="Path to an image file.")
    parser.add_argument("--prompt", default="Describe this image in one sentence.",
                        help="Prompt about the image.")
    parser.add_argument("--follow-up", default=None,
                        help="Optional second question, answered via cached image embeddings.")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    model = neat.genai.VisionLanguageModel(args.model)
    image = load_rgb_image(args.image)
    print(f"model_id={model.model_id()} accepts_image={model.accepts_image()} "
          f"image={image.shape} {image.dtype}")

    # Direct-image request.
    direct = neat.genai.GenerationRequest()
    direct.prompt = args.prompt
    direct.images = [image]
    direct.max_new_tokens = args.max_new_tokens
    print(f"direct: {model.run(direct).text}")

    # Optional follow-up reusing a cached image embedding.
    if args.follow_up:
        if not model.encode([image]):
            raise RuntimeError("VLM did not accept the image for caching; use direct images instead.")
        print(f"cached_images={model.cached_image_count()}")
        cached = neat.genai.GenerationRequest()
        cached.prompt = args.follow_up
        cached.use_cached_images = True
        cached.max_new_tokens = args.max_new_tokens
        print(f"cached: {model.run(cached).text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
