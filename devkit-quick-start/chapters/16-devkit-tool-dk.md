# Chapter 16 — The `dk` command

*Build in the SDK, run on the board, read the output where you are sitting.*

---

## What it does

`dk` (also installed as `devkit-run`) executes commands on the paired DevKit **from inside the SDK
container**, and streams the output back to your terminal. It is the piece that makes the
host-side workflow worth having: you never open a second terminal, never `scp` a binary, and never
guess which machine a log line came from.

```text
   SDK container                              Modalix DevKit
   ─────────────                              ──────────────
   $ dk build/my_app  ──────── runs it on ──────▶  ./my_app
        ▲                                             │
        └────────────── output streams back ──────────┘
```

Two things make it work: the **shared workspace** from [Chapter 15](15-install-neat-sdk.md), so the
file is already visible to both sides, and path translation inside `dk`, so file arguments resolve
correctly on the board even though the paths differ.

---

## Is a board actually paired?

Before anything else, ask:

```bash
sima-user@sdk:/workspace$ dk status
```

It reports the paired DevKit IP, the mounted path, and which sync method is active — NFS, or rsync
as a fallback. If it shows `SSH status : not reachable`, nothing else in this chapter will work;
pair the board again first:

```bash
sima-user@host:~$ sima-cli sdk setup --devkit <devkit-ip>
```

…pressing `n` at prompt 6 so you reuse the container you already built.

The other quick check is to simply open a shell on the board (below). If that lands you at a
`sima@modalix` prompt, the pairing is live.

---

## Running Python

```bash
sima-user@sdk:/workspace$ dk hello_neat.py
```

`dk` runs the script on the DevKit using the **DevKit's pyneat environment** — the one
[Chapter 11](11-install-pyneat.md) set up at `~/pyneat`. You do not activate anything first, and
you do not need pyneat installed inside the SDK container for this to work. The script executes on
the board; `print()` lands in your SDK terminal.

---

## Running a compiled C++ binary

Cross-compile in the SDK, then run the ARM64 binary on the board with the same command:

```bash
sima-user@sdk:/workspace$ dk build/sima_neat_hello
sima-user@sdk:/workspace$ dk build/my_app --config config.yaml
```

Arguments after the binary are passed straight through, including file paths — `dk` translates
them so the board resolves them correctly.

The script or binary must be **under `/workspace`** — `dk` refuses anything else, because that is
the folder the board can see.

The binary must be **ARM64**. A binary built for your x86 host will not run on the board; that is
what the SDK's cross-compilation toolchain is for. [Chapter 18](18-cpp-video-app.md) builds and
runs one end to end.

---

## Opening a shell on the board

```bash
sima-user@sdk:/workspace$ dk shell
sima@modalix:~$
```

An interactive session on the DevKit, from the SDK terminal. Use it for the things that are
awkward through single commands: poking at `/media/nvme`, tailing a log, checking `df -h` when an
install fails, running `simaai-sentinel` while an application works, or confirming a file actually
landed where the shared workspace said it would.

`exit` returns you to the SDK.

---

## Command summary

| Command | What it does |
|---|---|
| `dk status` | Paired DevKit IP, mounted path, sync method — and therefore whether pairing works |
| `dk <script>.py` | Runs a Python script on the board in the DevKit pyneat environment |
| `dk <binary> [args]` | Runs an ARM64 binary on the board, arguments and paths passed through |
| `dk shell` | Interactive shell on the DevKit |

`dk` is a shell function defined in `~/.devkit-sync.rc` and loaded from `~/.bashrc`, so it exists in
interactive SDK sessions. If the command is not found, source your profile — or you are not inside
the SDK container.

---

## Where to read more

- [DevKit Sync](https://developer.sima.ai/software/getting-started/dev-environment/devkit-sync#run-on-the-devkit-with-dk)
  — the reference for `dk` and the workspace sync model

---

| ← Previous | Contents | Next → |
|:---|:---:|---:|
| [Chapter 15 · Install the Neat SDK](15-install-neat-sdk.md) | [All chapters](../README.md) | [Chapter 17 · Neat Insight](17-neat-insight.md) |
