#!/usr/bin/env python3
"""Run pyneat.Model.benchmark() for one compiled model package."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_config(path: Path) -> dict:
    """Parse the house `key=value` config format.

    Deliberately dependency-free. This used to be YAML, which was a latent bug: the
    DevKit's python has no PyYAML, so `import yaml` raised there and the app could
    not read a config on the very board it is meant to run on. Every other app in
    this repo uses the same key=value `.conf`, so this is also the consistent choice.
    """
    config: dict = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_no}: expected key=value")
        key, value = (part.strip() for part in line.split("=", 1))
        config[key] = value
    return config


def spec_strings(model, method_name: str) -> list[str]:
    try:
        return [str(spec) for spec in getattr(model, method_name)()]
    except Exception as exc:
        return [f"unavailable: {exc}"]


def write_report(path: Path, model_path: Path, frames: int, model, report) -> None:
    data = {
        "benchmark": {
            "type": "model.synthetic",
            "frames": frames,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "model": {
            "path": str(model_path),
            "file": model_path.name,
            "input_specs": spec_strings(model, "input_specs"),
            "output_specs": spec_strings(model, "output_specs"),
        },
        "metrics": {
            "latency_ms": report.latency_ms,
            "fps": report.fps,
            "avg_power_watts": report.avg_power_watts,
            "energy_joules": report.energy_joules,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


APP_DIR = Path(__file__).resolve().parent


def resolve_app_path(value: str) -> Path:
    """Config paths are relative to THIS app folder, so the app works from any clone."""
    path = Path(str(value))
    return path if path.is_absolute() else APP_DIR / path


def main() -> int:
    # Config and outputs live inside this app, so it works from any clone location.
    default_config = APP_DIR / "config" / "default.conf"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--model", type=Path, help="compiled model package to benchmark")
    parser.add_argument("--frames", type=int, help="measured synthetic frames")
    parser.add_argument("--output-json", type=Path, help="benchmark report JSON path")
    args = parser.parse_args()

    try:
        needs_config = args.model is None or args.frames is None or args.output_json is None
        if args.config.is_file():
            config = load_config(args.config)
        elif needs_config:
            raise FileNotFoundError(f"config file does not exist: {args.config}")
        else:
            config = {}
        model_path = (
            args.model
            if args.model is not None
            else resolve_app_path(config.get("model_path", ""))
        )
        frames = (
            args.frames
            if args.frames is not None
            else int(config.get("frames", 1000))
        )
        report_path = (
            args.output_json
            if args.output_json is not None
            else resolve_app_path(config.get("output_json", "./output/report.json"))
        )
    except Exception as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if frames <= 0:
        print("benchmark.frames must be > 0", file=sys.stderr)
        return 2
    if not model_path.is_file():
        print(f"model file does not exist: {model_path}", file=sys.stderr)
        return 2

    try:
        import pyneat
    except ImportError:
        print("pyneat is not importable. Run: source ~/pyneat/bin/activate", file=sys.stderr)
        return 3

    try:
        model = pyneat.Model(str(model_path))
        report = model.benchmark(frames)
        write_report(report_path, model_path, frames, model, report)
    except Exception as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        return 4

    print(f"latency_ms={report.latency_ms}")
    print(f"fps={report.fps}")
    print(f"avg_power_watts={report.avg_power_watts}")
    print(f"energy_joules={report.energy_joules}")
    print(f"report_json={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
