#!/usr/bin/env bash
set -euo pipefail

source "${HOME}/.bashrc" >/dev/null 2>&1 || true
activate-model-compiler >/dev/null

echo "python=$(python --version)"
echo "python_path=$(command -v python)"

python - <<'PY'
mods = ["torch", "torchvision", "onnx", "onnxsim", "numpy", "PIL", "yaml"]
for name in mods:
    mod = __import__(name)
    print(f"{name}={getattr(mod, '__version__', 'ok')}")

try:
    import afe  # noqa: F401
    print("afe=ok")
except Exception as exc:
    print(f"afe=MISSING {exc}")

for optional in ["ultralytics", "timm", "pyneat"]:
    try:
        mod = __import__(optional)
        print(f"{optional}={getattr(mod, '__version__', 'ok')}")
    except Exception as exc:
        print(f"{optional}=MISSING {exc}")
PY
