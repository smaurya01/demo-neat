# Chapter 11 — Install pyneat

*The Neat Library and its Python API — what your applications actually call.*

---

## What gets installed

The Neat Library is the runtime that drives the accelerator. Installing it puts two things on the
board:

- **System packages** — `sima-neat` (the C++ library, `libsima_neat.so.6`), `sima-neat-dev`,
  `neat-codecs` and the runtime services underneath them
- **A Python virtual environment** at `~/pyneat`, containing the `pyneat` module

---

## Install

Log in first if you have not already this session:

*On the board*

```bash
sima-cli login
```

Then install the library, pinned to the version that matches board software 3.0.0:

*On the board*

```bash
cd /media/nvme && mkdir -p neat && cd neat
sima-cli neat install core@v0.6.0
```

> **Paired the board with the SDK already?** [Chapter 15](15-install-neat-sdk.md)'s setup installs
> the matching Neat Library and `~/pyneat` on the paired DevKit for you. Run the command above only
> to install on a board you use without the SDK, or to pin a version.

Run it from `/media/nvme/neat`: the installer downloads its packages into the current directory,
and your home directory is on the small eMMC.

If a selection screen appears, the useful extras are:

- **SiMa Neat extras**
- **Tutorials**
- **Tests**

While you are here, install the example applications too:

*On the board, in `/media/nvme/neat`*

```bash
sima-cli neat install apps@v0.6.0
```

> **Versions are a matched set.** Neat Apps 0.6.0 goes with Neat Library 0.6.0, which goes with
> board software 3.0.0 and Neat SDK 3.0.0.0. `sima-cli neat install` refuses a Neat Library that
> does not match the board software. Mixing them is the usual cause of "it built but it won't run" — see the
> [compatibility table](09-check-and-update-image.md#which-version-should-you-be-on).

---

## Verify

Check the installed packages:

*On the board*

```bash
dpkg -l | grep -E 'neat|sima-neat' | awk '{print $2, $3}'
```

```text
neat-codecs 0.6.0
neat-codecs-dev 0.6.0
sima-neat 0.6.0
sima-neat-dev 0.6.0
```

Every Neat package should read `0.6.0`. The C++ library is ABI 6 — applications built against an older Neat Library must be rebuilt:

*On the board*

```bash
ls /usr/lib/libsima_neat.so.*
```

```text
/usr/lib/libsima_neat.so.3.0.0
/usr/lib/libsima_neat.so.6
```

Then check the Python side:

*On the board*

```bash
source ~/pyneat/bin/activate
python3 -c "import pyneat; print(pyneat.__version__)"
```

```text
0.6.0
```

---

## Using it

`pyneat` lives in a virtual environment, so **activate it first** in every new shell:

*On the board*

```bash
source ~/pyneat/bin/activate
```

Or call the interpreter directly, which is more reliable in scripts and over SSH:

*On the board*

```bash
/home/sima/pyneat/bin/python my_app.py
```

Running a script over SSH without activating the environment is a common trip-up:

*On the host*

```bash
ssh sima@<devkit-ip> 'source $HOME/pyneat/bin/activate; python /media/nvme/app.py'
```

---

## Explore the API

*On the board, with `~/pyneat` activated*

```bash
python3 -c "import pyneat; print([n for n in dir(pyneat) if not n.startswith('_')])"
python3 -c "import pyneat; help(pyneat.ModelOptions)"
```

The two objects you will meet first are **`Model`** — load a compiled archive and run it — and
**`Graph`** — wire several stages together into a pipeline. [Chapter 12](12-object-detection.md)
uses `Model`.

Full reference: [Python API](https://developer.sima.ai/software/reference/pythonapi/) ·
[C++ API](https://developer.sima.ai/software/reference/cppapi/)

---

| ← Previous | Contents | Next → |
|:---|:---:|---:|
| [Chapter 10 · Install simaai-sentinel](10-install-simaai-sentinel.md) | [All chapters](../README.md) | [Chapter 12 · Object detection on images](12-object-detection.md) |
