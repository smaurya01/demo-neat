# Chapter 13 — Object detection on images

*Your first real inference: a folder of images in, annotated images out.*

> **This chapter is a work in progress.** The application code will be added here. Everything below
> — the model, the prerequisites and the verification steps — is ready and correct, so you can set
> up now and drop the code in when it lands.

---

## Before you start

| Requirement | Chapter |
|---|---|
| Board on the network with internet | [5](05-internet-sharing.md) · [6](06-connect-to-router.md) · [7](07-static-connection.md) |
| `sima-cli` installed and logged in | [8](08-install-sima-cli.md) |
| NVMe mounted at `/media/nvme` | [9](09-mount-nvme.md) |
| Board software 2.1.3 | [10](10-check-and-update-image.md) |
| `pyneat` installed | [12](12-install-pyneat.md) |

---

## Get a compiled model

Models for the MLA are distributed as compiled `.tar.gz` archives. Browse what is available:

```bash
sima@modalix:~$ sima-cli modelzoo
```

Download into the NVMe, not your home directory:

```bash
sima@modalix:~$ mkdir -p /media/nvme/models
sima@modalix:~$ cd /media/nvme/models
sima@modalix:~$ sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/yolo26-detection/<model-file>"
```

Set the Model Zoo release first — it can differ from your board software version:

```bash
sima@modalix:~$ export MODELZOO_VERSION="2.1.3"
```

A good default for detection is a YOLO26 BF16 build with MLA tessellation, which is what the
example applications use.

---

## Prepare some input images

```bash
sima@modalix:~$ mkdir -p /media/nvme/images
```

Copy in a handful of JPEGs or PNGs. Anything with recognisable objects — people, vehicles, laptops,
cups — works for a COCO-trained detector.

---

## The application

> **Code to be added.**
>
> It will go here, along with a `config.yaml` and the exact command to run it.

<!-- TODO: application code, config, and run command -->

---

## Run it

> **To be completed with the code above.**

The shape it will take:

```bash
sima@modalix:~$ source ~/pyneat/bin/activate
(pyneat) sima@modalix:~$ python3 detect.py --config config.yaml
```

---

## What you should see

A per-image line naming the detections found, then one annotated image written per input:

```text
[1/5] image_01.jpg -> image_01.png (3 detections)
...
Done: 5/5 images
```

Open the output images to confirm the boxes land on the right objects.

---

## If it is slower than you expect

Run `simaai-sentinel` ([Chapter 11](11-install-simaai-sentinel.md)) in a second terminal while it
works. For single-image inference the usual finding is that the MLA is barely busy and the time
goes on image decode and PNG encode — inference itself is often a small fraction of wall-clock time.
That is normal and not a problem to fix.

---

## Where to go next

| Next step | Where |
|---|---|
| Ready-to-run example applications | `sima-cli neat install apps@v0.5.0`, then browse `prebuilt-apps/examples/` |
| Run a language model on the board | [Chapter 14](14-llima.md) |
| The full tutorial series | [Tutorials](https://developer.sima.ai/software/tutorials) |
| Compile your own ONNX model | [Model Compiler](https://developer.sima.ai/software/getting-started/) |

---

| ← Previous | Contents | Next → |
|:---|:---:|---:|
| [Chapter 12 · Install pyneat](12-install-pyneat.md) | [All chapters](README.md) | [Chapter 14 · LLiMa](14-llima.md) |
