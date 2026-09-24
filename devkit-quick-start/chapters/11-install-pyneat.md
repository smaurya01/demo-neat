# Chapter 11 — Install pyneat

*The Neat Library and its Python API — what your applications actually call.*

---

## What gets installed

The Neat Library is the runtime that drives the accelerator. Installing it puts two things on the
board:

- **System packages** — `neat-runtime`, `neat-appcomplex`, `neat-gst-plugins`, `sima-neat` and
  friends, which are the C++ library and the GStreamer elements underneath it
- **A Python virtual environment** at `~/pyneat`, containing the `pyneat` module

---

## Install

Log in first if you have not already this session:

```bash
sima@modalix:~$ sima-cli login
```

Then install the library, pinned to the version that matches board software 2.1.3:

```bash
sima@modalix:~$ cd /media/nvme && mkdir -p neat && cd neat
sima@modalix:/media/nvme/neat$ sima-cli neat install core@v0.4.0
```

Run it from `/media/nvme/neat`: the installer downloads its packages into the current directory,
and your home directory is on the small eMMC.

If a selection screen appears, the useful extras are:

- **SiMa Neat extras**
- **Tutorials**
- **Tests**

While you are here, install the example applications too:

```bash
sima@modalix:~$ sima-cli neat install apps@v0.5.0
```

> **Versions are a matched set.** Neat Apps 0.5.0 goes with Neat Library 0.4.0, which goes with
> board software 2.1.3. Mixing them is the usual cause of "it built but it won't run" — see the
> [compatibility table](09-check-and-update-image.md#which-version-should-you-be-on).

---

## Verify

Check the installed packages:

```bash
sima@modalix:~$ dpkg -l | grep -E 'neat|sima-neat' | awk '{print $2, $3}'
```

Expect `0.4.0` against `neat-runtime`, `neat-appcomplex`, `neat-common`, `sima-neat` and the rest.

Then check the Python side:

```bash
sima@modalix:~$ source ~/pyneat/bin/activate
(pyneat) sima@modalix:~$ python3 -c "import pyneat; print('pyneat OK')"
```

---

## Using it

`pyneat` lives in a virtual environment, so **activate it first** in every new shell:

```bash
sima@modalix:~$ source ~/pyneat/bin/activate
```

Or call the interpreter directly, which is more reliable in scripts and over SSH:

```bash
sima@modalix:~$ /home/sima/pyneat/bin/python my_app.py
```

Running a script over SSH without activating the environment is a common trip-up:

```bash
sima-user@host:~$ ssh sima@<devkit-ip> 'source $HOME/pyneat/bin/activate; python /media/nvme/app.py'
```

---

## Explore the API

```bash
(pyneat) sima@modalix:~$ python3 -c "import pyneat; print([n for n in dir(pyneat) if not n.startswith('_')])"
(pyneat) sima@modalix:~$ python3 -c "import pyneat; help(pyneat.ModelOptions)"
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
