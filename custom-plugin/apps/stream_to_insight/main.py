"""stream_to_insight (Python): the same pipeline as main.cpp, built with pyneat.

    RTSP (from Insight) -> H.264 decode -> framemarker -> branch -+-> H.264 encode -> Insight
                                                                  +-> "stats" (pulled here)

Python cannot subclass a Neat Node, so the plugin goes in with pyneat.nodes.custom(), which
takes the same launch fragment you would give gst-launch-1.0.

Usage:
    python3 main.py --url rtsp://<insight-host>:8554/src1 --insight-host <insight-host> \
        [--channel 0] [--mode grayscale] [--border 12] [--color 0x00FF00] [--no-bar] \
        [--log-interval N]

Without --width/--height/--fps the stream is probed with OpenCV (cv2).
"""

import argparse
import signal
import time

import pyneat


def probe_stream(url):
    """Return (width, height, fps) of the RTSP stream using OpenCV."""
    import cv2  # only needed when --width/--height/--fps are not given

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {url} to probe its size and frame rate")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    cap.release()
    if w <= 0 or h <= 0 or fps <= 0:
        raise RuntimeError("could not probe stream geometry; pass --width --height --fps")
    return w, h, fps


def framemarker_fragment(args):
    """The element and its properties, exactly as gst-launch-1.0 would take them."""
    return (
        f"framemarker name=marker mode={args.mode} border-width={args.border} "
        f"color=0x{args.color:06X} progress-bar={'false' if args.no_bar else 'true'} "
        f"log-interval={args.log_interval}"
    )


def color_value(text):
    """argparse type for --color: decimal or 0xRRGGBB."""
    try:
        value = int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected 0xRRGGBB, got '{text}'")
    if not 0 <= value <= 0xFFFFFF:
        raise argparse.ArgumentTypeError(f"must be 0x000000..0xFFFFFF, got '{text}'")
    return value


def main():
    # Install the stop handlers first so a Ctrl-C during start-up (probe, build) still ends in
    # run.close() instead of killing the process half-way.
    stop = False

    def on_signal(*_):
        nonlocal stop
        stop = True

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, on_signal)

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--url", required=True)
    p.add_argument("--insight-host", default="127.0.0.1")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--video-port-base", type=int, default=9000)
    p.add_argument("--mode", choices=["none", "grayscale", "invert"], default="grayscale")
    p.add_argument("--border", type=int, default=12, help="border width in pixels, 0..256")
    p.add_argument("--color", type=color_value, default="0x00FF00", help="border colour 0xRRGGBB")
    p.add_argument("--no-bar", action="store_true")
    p.add_argument("--log-interval", type=int, default=0,
                   help="plugin prints frames/fps every N frames (0 = off)")
    p.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = run forever)")
    p.add_argument("--width", type=int, default=0)
    p.add_argument("--height", type=int, default=0)
    p.add_argument("--fps", type=int, default=0)
    p.add_argument("--print-graph", action="store_true")
    args = p.parse_args()
    if not 0 <= args.border <= 256:
        p.error(f"--border must be between 0 and 256, got {args.border}")
    if args.log_interval < 0:
        p.error(f"--log-interval must be 0 or more, got {args.log_interval}")

    if args.width > 0 and args.height > 0 and args.fps > 0:
        width, height, fps = args.width, args.height, args.fps
    else:
        width, height, fps = probe_stream(args.url)
    if stop:
        return

    # Source: RTSP -> depay -> parse -> hardware decode, pinned to NV12 at the stream size.
    src = pyneat.RtspDecodedInputOptions()
    src.url = args.url
    src.tcp = True
    src.latency_ms = 200
    src.codec = pyneat.RtspCodec.H264
    src.payload_type = 96
    src.source_fps = fps
    src.fallback_h264_width = width
    src.fallback_h264_height = height
    caps = src.output_caps
    caps.enable = True
    caps.format = pyneat.Format.NV12
    caps.width = width
    caps.height = height
    caps.fps = fps
    caps.memory = pyneat.CapsMemory.Any
    src.output_caps = caps
    source = pyneat.groups.rtsp_decoded_input(src)

    # The custom plugin.
    marker = pyneat.nodes.custom(framemarker_fragment(args))

    # Video out to Insight.
    sender = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(width, height, fps)
    sender.host = args.insight_host
    sender.channel = args.channel
    sender.video_port_base = args.video_port_base
    sender.encoder.bitrate_kbps = 2000
    video = pyneat.Graph("video")
    video.connect(pyneat.nodes.input("video"), pyneat.groups.video_sender(sender))

    stats = pyneat.Graph("stats")
    stats.add(pyneat.nodes.output("stats", pyneat.OutputOptions.latest()))

    graph = pyneat.Graph("stream_to_insight")
    branch = pyneat.graphs.branch("marked", ["video", "stats"])
    graph.connect(source, marker)
    graph.connect(marker, branch)
    graph.connect(branch, video)
    graph.connect(branch, stats)

    if args.print_graph:
        print(graph.describe())
        print("Backend:\n" + graph.describe_backend())

    opt = pyneat.RunOptions()
    opt.preset = pyneat.RunPreset.Realtime
    opt.queue_depth = 3
    opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    opt.output_memory = pyneat.OutputMemory.ZeroCopy
    run = graph.build(opt)

    print(
        f"source={args.url} {width}x{height}@{fps} mode={args.mode} "
        f"insight={args.insight_host}:{sender.video_port} channel={args.channel}\n"
        "Press Ctrl-C to stop.",
        flush=True,
    )

    # A Ctrl-C that arrived during build() skips the loop and goes straight to close().
    total, window, window_start = 0, 0, time.monotonic()
    try:
        while not stop and (args.frames <= 0 or total < args.frames):
            sample = run.pull("stats", 1000)
            if sample is None:
                # pull() returns None on a timeout and when the stream has ended; only the
                # second one stops the run.
                if not run.can_pull():
                    break
                continue
            total += 1
            window += 1
            secs = time.monotonic() - window_start
            if secs >= 5.0:
                print(
                    f"frames={total} pulled_fps={window / secs:.1f} "
                    f"last_frame_id={sample.frame_id}",
                    flush=True,
                )
                window, window_start = 0, time.monotonic()
    finally:
        run.close()
    print(f"stopped after {total} frames")


if __name__ == "__main__":
    main()
