/*
 * framemarker: a sample GStreamer video filter for SiMa Neat pipelines.
 *
 * It works in place on NV12 or I420 frames, which is what the Neat decoders produce, and does
 * three things so that its effect is easy to see in Neat Insight:
 *
 *   mode          none | grayscale | invert   -- whole-frame pixel operation
 *   border-width  N pixels                    -- solid border in `color`
 *   progress-bar  true | false                -- a bar that grows by one step per frame
 *
 * Every property can be changed while the pipeline is playing.
 *
 * Build: see ../README.md, or "Step 3" of ../CUSTOM_PLUGIN_TUTORIAL.md. The resulting file is
 * libgstframemarker.so and the element name is `framemarker`.
 */

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>

#include <string.h>

GST_DEBUG_CATEGORY_STATIC(gst_frame_marker_debug);
#define GST_CAT_DEFAULT gst_frame_marker_debug

/* ------------------------------------------------------------------------------------------ */
/* The `mode` enum property                                                                    */
/* ------------------------------------------------------------------------------------------ */

typedef enum {
  FRAME_MARKER_MODE_NONE = 0,
  FRAME_MARKER_MODE_GRAYSCALE = 1,
  FRAME_MARKER_MODE_INVERT = 2,
} GstFrameMarkerMode;

#define GST_TYPE_FRAME_MARKER_MODE (gst_frame_marker_mode_get_type())
static GType gst_frame_marker_mode_get_type(void) {
  static gsize type_id = 0;
  static const GEnumValue values[] = {
      {FRAME_MARKER_MODE_NONE, "Leave pixels unchanged", "none"},
      {FRAME_MARKER_MODE_GRAYSCALE, "Remove colour (chroma = 128)", "grayscale"},
      {FRAME_MARKER_MODE_INVERT, "Invert luma and chroma", "invert"},
      {0, NULL, NULL},
  };
  if (g_once_init_enter(&type_id)) {
    GType t = g_enum_register_static("GstFrameMarkerMode", values);
    g_once_init_leave(&type_id, t);
  }
  return (GType)type_id;
}

/* ------------------------------------------------------------------------------------------ */
/* Element instance                                                                            */
/* ------------------------------------------------------------------------------------------ */

#define GST_TYPE_FRAME_MARKER (gst_frame_marker_get_type())
G_DECLARE_FINAL_TYPE(GstFrameMarker, gst_frame_marker, GST, FRAME_MARKER, GstVideoFilter)

struct _GstFrameMarker {
  GstVideoFilter parent;

  /* Properties. Written from the application thread, read from the streaming thread, so both
   * sides take the object lock. */
  GstFrameMarkerMode mode;
  guint border_width;
  guint color; /* 0xRRGGBB */
  gboolean progress_bar;
  guint log_interval;

  /* Streaming-thread state. */
  guint64 frames;
  gint64 window_start_us;
  guint64 window_start_frame;
};

G_DEFINE_TYPE(GstFrameMarker, gst_frame_marker, GST_TYPE_VIDEO_FILTER)

enum {
  PROP_0,
  PROP_MODE,
  PROP_BORDER_WIDTH,
  PROP_COLOR,
  PROP_PROGRESS_BAR,
  PROP_LOG_INTERVAL,
};

#define DEFAULT_MODE FRAME_MARKER_MODE_NONE
#define DEFAULT_BORDER_WIDTH 8
#define DEFAULT_COLOR 0x00FF00
#define DEFAULT_PROGRESS_BAR TRUE
#define DEFAULT_LOG_INTERVAL 0
#define PROGRESS_STEPS 120

/* Both pads accept exactly what the Neat decoders emit: raw NV12 or I420 in system-mappable
 * memory. Caps negotiation fails at Graph::build() if the upstream element offers anything else. */
#define FRAME_MARKER_CAPS GST_VIDEO_CAPS_MAKE("{ NV12, I420 }")

static GstStaticPadTemplate sink_template =
    GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS(FRAME_MARKER_CAPS));
static GstStaticPadTemplate src_template =
    GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS(FRAME_MARKER_CAPS));

/* ------------------------------------------------------------------------------------------ */
/* Pixel helpers                                                                               */
/* ------------------------------------------------------------------------------------------ */

typedef struct {
  guint8 y, u, v;
} Yuv;

/* BT.601 limited range, which is what the H.264 encoder downstream assumes. */
static Yuv rgb_to_yuv(guint rgb) {
  const gint r = (rgb >> 16) & 0xFF, g = (rgb >> 8) & 0xFF, b = rgb & 0xFF;
  Yuv out;
  out.y = (guint8)CLAMP(((66 * r + 129 * g + 25 * b + 128) >> 8) + 16, 0, 255);
  out.u = (guint8)CLAMP(((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128, 0, 255);
  out.v = (guint8)CLAMP(((112 * r - 94 * g - 18 * b + 128) >> 8) + 128, 0, 255);
  return out;
}

/* Fill a rectangle in frame coordinates. Chroma is 2x2 subsampled in both NV12 and I420; NV12
 * keeps U and V interleaved in plane 1, I420 keeps them in planes 1 and 2. Always use the
 * per-plane stride: rows are often padded beyond the visible width. */
static void fill_rect(GstVideoFrame *f, gint x, gint y, gint w, gint h, Yuv c) {
  const gint fw = GST_VIDEO_FRAME_WIDTH(f), fh = GST_VIDEO_FRAME_HEIGHT(f);
  const gint x0 = MAX(x, 0), y0 = MAX(y, 0);
  const gint x1 = MIN(x + w, fw), y1 = MIN(y + h, fh);
  if (x0 >= x1 || y0 >= y1)
    return;

  guint8 *luma = GST_VIDEO_FRAME_PLANE_DATA(f, 0);
  const gint luma_stride = GST_VIDEO_FRAME_PLANE_STRIDE(f, 0);
  for (gint row = y0; row < y1; ++row)
    memset(luma + (gsize)row * luma_stride + x0, c.y, x1 - x0);

  const gint cx0 = x0 / 2, cy0 = y0 / 2, cx1 = (x1 + 1) / 2, cy1 = (y1 + 1) / 2;
  if (GST_VIDEO_FRAME_FORMAT(f) == GST_VIDEO_FORMAT_NV12) {
    guint8 *uv = GST_VIDEO_FRAME_PLANE_DATA(f, 1);
    const gint stride = GST_VIDEO_FRAME_PLANE_STRIDE(f, 1);
    for (gint row = cy0; row < cy1; ++row) {
      guint8 *p = uv + (gsize)row * stride;
      for (gint col = cx0; col < cx1; ++col) {
        p[2 * col] = c.u;
        p[2 * col + 1] = c.v;
      }
    }
  } else { /* I420 */
    guint8 *u = GST_VIDEO_FRAME_PLANE_DATA(f, 1), *v = GST_VIDEO_FRAME_PLANE_DATA(f, 2);
    const gint us = GST_VIDEO_FRAME_PLANE_STRIDE(f, 1), vs = GST_VIDEO_FRAME_PLANE_STRIDE(f, 2);
    for (gint row = cy0; row < cy1; ++row) {
      memset(u + (gsize)row * us + cx0, c.u, cx1 - cx0);
      memset(v + (gsize)row * vs + cx0, c.v, cx1 - cx0);
    }
  }
}

/* Apply `fn` to every visible byte of one plane. */
static void for_each_plane_byte(GstVideoFrame *f, guint plane, guint8 (*fn)(guint8)) {
  guint8 *data = GST_VIDEO_FRAME_PLANE_DATA(f, plane);
  const gint stride = GST_VIDEO_FRAME_PLANE_STRIDE(f, plane);
  /* Visible bytes per row: component width times the number of components packed in the
   * plane (2 for NV12's interleaved UV, otherwise 1). */
  const guint comp = (plane == 0) ? 0 : 1;
  const gint rows = GST_VIDEO_FRAME_COMP_HEIGHT(f, comp);
  gint row_bytes = GST_VIDEO_FRAME_COMP_WIDTH(f, comp);
  if (plane == 1 && GST_VIDEO_FRAME_FORMAT(f) == GST_VIDEO_FORMAT_NV12)
    row_bytes *= 2;
  for (gint row = 0; row < rows; ++row) {
    guint8 *p = data + (gsize)row * stride;
    for (gint i = 0; i < row_bytes; ++i)
      p[i] = fn(p[i]);
  }
}

static guint8 to_neutral(guint8 v) {
  (void)v;
  return 128;
}
static guint8 invert(guint8 v) {
  return (guint8)(255 - v);
}

/* ------------------------------------------------------------------------------------------ */
/* GstVideoFilter virtual methods                                                              */
/* ------------------------------------------------------------------------------------------ */

static gboolean gst_frame_marker_start(GstBaseTransform *trans) {
  GstFrameMarker *self = GST_FRAME_MARKER(trans);
  self->frames = 0;
  self->window_start_us = g_get_monotonic_time();
  self->window_start_frame = 0;
  return TRUE;
}

/* Called once caps are negotiated, before the first buffer. */
static gboolean gst_frame_marker_set_info(GstVideoFilter *filter, GstCaps *incaps,
                                          GstVideoInfo *in_info, GstCaps *outcaps,
                                          GstVideoInfo *out_info) {
  (void)incaps;
  (void)outcaps;
  (void)out_info;
  GST_INFO_OBJECT(filter, "negotiated %s %dx%d @ %d/%d",
                  gst_video_format_to_string(GST_VIDEO_INFO_FORMAT(in_info)),
                  GST_VIDEO_INFO_WIDTH(in_info), GST_VIDEO_INFO_HEIGHT(in_info),
                  GST_VIDEO_INFO_FPS_N(in_info), GST_VIDEO_INFO_FPS_D(in_info));
  return TRUE;
}

/* Called for every frame. The frame is already mapped read-write; timestamps and all other
 * buffer metadata pass through untouched, which is what keeps Insight's video/metadata
 * correlation working downstream. */
static GstFlowReturn gst_frame_marker_transform_frame_ip(GstVideoFilter *filter,
                                                         GstVideoFrame *frame) {
  GstFrameMarker *self = GST_FRAME_MARKER(filter);

  GST_OBJECT_LOCK(self);
  const GstFrameMarkerMode mode = self->mode;
  const gint bw = (gint)self->border_width;
  const Yuv color = rgb_to_yuv(self->color);
  const gboolean bar = self->progress_bar;
  const guint log_interval = self->log_interval;
  GST_OBJECT_UNLOCK(self);

  const gint w = GST_VIDEO_FRAME_WIDTH(frame), h = GST_VIDEO_FRAME_HEIGHT(frame);
  const guint n_planes = GST_VIDEO_FRAME_N_PLANES(frame);

  switch (mode) {
  case FRAME_MARKER_MODE_GRAYSCALE:
    for (guint p = 1; p < n_planes; ++p)
      for_each_plane_byte(frame, p, to_neutral);
    break;
  case FRAME_MARKER_MODE_INVERT:
    for (guint p = 0; p < n_planes; ++p)
      for_each_plane_byte(frame, p, invert);
    break;
  case FRAME_MARKER_MODE_NONE:
  default:
    break;
  }

  if (bw > 0) {
    fill_rect(frame, 0, 0, w, bw, color);          /* top */
    fill_rect(frame, 0, h - bw, w, bw, color);     /* bottom */
    fill_rect(frame, 0, 0, bw, h, color);          /* left */
    fill_rect(frame, w - bw, 0, bw, h, color);     /* right */
  }

  if (bar) {
    const Yuv white = {235, 128, 128};
    const gint bar_h = MAX(4, h / 40);
    const gint inner_w = MAX(0, w - 2 * bw);
    const gint step = (gint)(self->frames % PROGRESS_STEPS) + 1;
    fill_rect(frame, bw, h - bw - bar_h, inner_w * step / PROGRESS_STEPS, bar_h, white);
  }

  self->frames++;
  GST_LOG_OBJECT(self, "frame %" G_GUINT64_FORMAT " pts %" GST_TIME_FORMAT, self->frames,
                 GST_TIME_ARGS(GST_BUFFER_PTS(frame->buffer)));

  if (log_interval > 0 && self->frames % log_interval == 0) {
    const gint64 now = g_get_monotonic_time();
    const double secs = (now - self->window_start_us) / 1e6;
    const double fps = secs > 0 ? (self->frames - self->window_start_frame) / secs : 0.0;
    g_print("[framemarker %s] frames=%" G_GUINT64_FORMAT " fps=%.1f %dx%d %s\n",
            GST_OBJECT_NAME(self), self->frames, fps, w, h,
            gst_video_format_to_string(GST_VIDEO_FRAME_FORMAT(frame)));
    self->window_start_us = now;
    self->window_start_frame = self->frames;
  }

  return GST_FLOW_OK;
}

/* ------------------------------------------------------------------------------------------ */
/* Properties                                                                                  */
/* ------------------------------------------------------------------------------------------ */

static void gst_frame_marker_set_property(GObject *object, guint prop_id, const GValue *value,
                                          GParamSpec *pspec) {
  GstFrameMarker *self = GST_FRAME_MARKER(object);
  GST_OBJECT_LOCK(self);
  switch (prop_id) {
  case PROP_MODE:
    self->mode = g_value_get_enum(value);
    break;
  case PROP_BORDER_WIDTH:
    self->border_width = g_value_get_uint(value);
    break;
  case PROP_COLOR:
    self->color = g_value_get_uint(value) & 0xFFFFFF;
    break;
  case PROP_PROGRESS_BAR:
    self->progress_bar = g_value_get_boolean(value);
    break;
  case PROP_LOG_INTERVAL:
    self->log_interval = g_value_get_uint(value);
    break;
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
    break;
  }
  GST_OBJECT_UNLOCK(self);
}

static void gst_frame_marker_get_property(GObject *object, guint prop_id, GValue *value,
                                          GParamSpec *pspec) {
  GstFrameMarker *self = GST_FRAME_MARKER(object);
  GST_OBJECT_LOCK(self);
  switch (prop_id) {
  case PROP_MODE:
    g_value_set_enum(value, self->mode);
    break;
  case PROP_BORDER_WIDTH:
    g_value_set_uint(value, self->border_width);
    break;
  case PROP_COLOR:
    g_value_set_uint(value, self->color);
    break;
  case PROP_PROGRESS_BAR:
    g_value_set_boolean(value, self->progress_bar);
    break;
  case PROP_LOG_INTERVAL:
    g_value_set_uint(value, self->log_interval);
    break;
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
    break;
  }
  GST_OBJECT_UNLOCK(self);
}

/* ------------------------------------------------------------------------------------------ */
/* Type registration                                                                           */
/* ------------------------------------------------------------------------------------------ */

static void gst_frame_marker_class_init(GstFrameMarkerClass *klass) {
  GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
  GstBaseTransformClass *trans_class = GST_BASE_TRANSFORM_CLASS(klass);
  GstVideoFilterClass *filter_class = GST_VIDEO_FILTER_CLASS(klass);

  gobject_class->set_property = gst_frame_marker_set_property;
  gobject_class->get_property = gst_frame_marker_get_property;

  const GParamFlags flags =
      G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_CONTROLLABLE;
  g_object_class_install_property(
      gobject_class, PROP_MODE,
      g_param_spec_enum("mode", "Mode", "Whole-frame pixel operation", GST_TYPE_FRAME_MARKER_MODE,
                        DEFAULT_MODE, flags));
  g_object_class_install_property(
      gobject_class, PROP_BORDER_WIDTH,
      g_param_spec_uint("border-width", "Border width", "Border thickness in pixels (0 = off)", 0,
                        256, DEFAULT_BORDER_WIDTH, flags));
  g_object_class_install_property(
      gobject_class, PROP_COLOR,
      g_param_spec_uint("color", "Color", "Border colour as 0xRRGGBB", 0, 0xFFFFFF, DEFAULT_COLOR,
                        flags));
  g_object_class_install_property(
      gobject_class, PROP_PROGRESS_BAR,
      g_param_spec_boolean("progress-bar", "Progress bar",
                           "Draw a bar that advances one step per frame", DEFAULT_PROGRESS_BAR,
                           flags));
  g_object_class_install_property(
      gobject_class, PROP_LOG_INTERVAL,
      g_param_spec_uint("log-interval", "Log interval",
                        "Print frame count and fps every N frames (0 = off)", 0, G_MAXUINT,
                        DEFAULT_LOG_INTERVAL, flags));

  gst_element_class_add_static_pad_template(element_class, &sink_template);
  gst_element_class_add_static_pad_template(element_class, &src_template);
  gst_element_class_set_static_metadata(
      element_class, "Frame marker", "Filter/Effect/Video",
      "Sample custom filter: grayscale/invert, coloured border and a per-frame progress bar",
      "demo-neat custom-plugin tutorial");

  trans_class->start = GST_DEBUG_FUNCPTR(gst_frame_marker_start);
  filter_class->set_info = GST_DEBUG_FUNCPTR(gst_frame_marker_set_info);
  filter_class->transform_frame_ip = GST_DEBUG_FUNCPTR(gst_frame_marker_transform_frame_ip);

  gst_type_mark_as_plugin_api(GST_TYPE_FRAME_MARKER_MODE, 0);
}

static void gst_frame_marker_init(GstFrameMarker *self) {
  self->mode = DEFAULT_MODE;
  self->border_width = DEFAULT_BORDER_WIDTH;
  self->color = DEFAULT_COLOR;
  self->progress_bar = DEFAULT_PROGRESS_BAR;
  self->log_interval = DEFAULT_LOG_INTERVAL;
}

static gboolean plugin_init(GstPlugin *plugin) {
  GST_DEBUG_CATEGORY_INIT(gst_frame_marker_debug, "framemarker", 0, "Frame marker sample filter");
  return gst_element_register(plugin, "framemarker", GST_RANK_NONE, GST_TYPE_FRAME_MARKER);
}

#ifndef PACKAGE
#define PACKAGE "framemarker"
#endif

/* The first name must match the file name: libgst<name>.so. */
GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, framemarker,
                  "Sample custom video filter for SiMa Neat pipelines", plugin_init, "1.0.0",
                  "LGPL", PACKAGE, "https://gstreamer.freedesktop.org")
