# NEAT Component Overview

What each piece of the NEAT stack is, and where it runs. For setup steps, see
[`README.md`](README.md).

- **Host:** Development machine with `sima-cli`, container runtime, and local workspace.
- **Modalix DevKit (Board):** Target hardware running Modalix firmware where applications execute.
- **NEAT SDK (cross-compile):** Containerized environment for building C++ apps, preparing model artifacts, and pairing with the board.
- **Neat Core:** Runtime C++ libraries that power model execution and app APIs on Modalix.
- **PyNeat:** Python bindings/runtime for prototyping and running NEAT apps on the DevKit.
- **Model Compiler:** Optional toolchain to compile/quantize ONNX or GenAI models for Modalix.
- **NEAT Insight:** Browser-based inspection and debugging tool for runtime streams, files, and logs. See [neat_insight.md](neat_insight.md).
- **NEAT Apps:** User applications built with NEAT C++ or PyNeat deployed to the DevKit.

<img src="https://developer.sima.ai/software/assets/images/neat-software-stack-animated-cdd17bb3f6d7e02b6b2742cdf649b6bf.svg" alt="NEAT software stack diagram: host, SDK container, and Modalix DevKit" width="900">
