#!/usr/bin/env python3
import argparse
import json
import tarfile
from pathlib import Path


def looks_like_elf(tar, member):
    if not member.isfile():
        return False
    f = tar.extractfile(member)
    if f is None:
        return False
    return f.read(4) == b"\x7fELF"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate a compiled .tar.gz MPK archive. Default (no flags) keeps the "
            "STRICT T1/T5 YOLO/CNN contract: exactly one ELF, zero .so. The T7 "
            "transformer/difficult-model policy is opt-in via --max-elf/--allow-so."
        )
    )
    parser.add_argument("--archive", required=True)
    parser.add_argument("--report", default=None)
    # T7 sanctioned extension. Defaults reproduce the original strict behaviour.
    parser.add_argument("--min-elf", type=int, default=1,
                        help="minimum acceptable ELF members (default 1)")
    parser.add_argument("--max-elf", type=int, default=1,
                        help="maximum acceptable ELF members (default 1 = strict). T7 uses 3.")
    parser.add_argument("--allow-so", action="store_true",
                        help="T7 relaxed mode: report .so members as 'requires_justification' "
                             "instead of an automatic fail (strict mode still fails on any .so).")
    args = parser.parse_args()

    archive = Path(args.archive)
    result = {
        "archive": str(archive),
        "exists": archive.exists(),
        "is_tar_gz": archive.suffixes[-2:] == [".tar", ".gz"] or archive.name.endswith(".tgz"),
        "members": [],
        "elf_members": [],
        "so_members": [],
        "policy": {"min_elf": args.min_elf, "max_elf": args.max_elf, "allow_so": args.allow_so},
        "status": "fail",
    }
    if result["exists"]:
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                result["members"].append(member.name)
                is_so = member.name.endswith(".so") or ".so." in member.name
                if is_so:
                    result["so_members"].append(member.name)
                # NOTE: .so shared objects also carry the \x7fELF magic, so they must
                # be excluded from the ELF-stage count or a fragmented archive is
                # wildly over-counted (a 58-elf/78-so archive read as 136 "elf").
                elif member.name.endswith(".elf") or looks_like_elf(tar, member):
                    result["elf_members"].append(member.name)

    n_elf = len(result["elf_members"])
    n_so = len(result["so_members"])
    # Backward-compatible strict signals (kept for existing callers/reports).
    result["single_elf"] = n_elf == 1
    result["no_so"] = n_so == 0
    # Policy-aware signals.
    result["elf_in_range"] = args.min_elf <= n_elf <= args.max_elf
    # In relaxed mode a .so is not an automatic fail but MUST be justified in the
    # model's surgery report; we surface that here rather than silently passing.
    result["so_requires_justification"] = args.allow_so and n_so > 0
    so_ok = result["no_so"] or args.allow_so

    base_ok = result["exists"] and result["is_tar_gz"] and result["elf_in_range"] and so_ok
    if base_ok:
        # An out-of-range ELF count already failed base_ok, so we only reach here
        # when the ELF count is acceptable; a justified .so downgrades to a
        # "needs a written reason" pass, never a clean pass.
        result["status"] = "pass_requires_justification" if result["so_requires_justification"] else "pass"

    output = json.dumps(result, indent=2)
    if args.report:
        Path(args.report).write_text(output + "\n", encoding="utf-8")
    print(output)
    raise SystemExit(0 if result["status"].startswith("pass") else 1)


if __name__ == "__main__":
    main()
