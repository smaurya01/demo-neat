#include "utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace neat = simaai::neat;

namespace {

struct Nv12Color {
  std::uint8_t y = 235;
  std::uint8_t u = 128;
  std::uint8_t v = 128;
};

Nv12Color pose_color(int index) {
  static constexpr std::array<Nv12Color, 12> palette = {
      Nv12Color{82, 90, 240},   Nv12Color{170, 60, 180}, Nv12Color{210, 40, 40},
      Nv12Color{180, 40, 150},  Nv12Color{200, 100, 30}, Nv12Color{145, 54, 34},
      Nv12Color{160, 210, 190}, Nv12Color{120, 220, 90}, Nv12Color{225, 110, 170},
      Nv12Color{190, 80, 210},  Nv12Color{235, 128, 128}, Nv12Color{95, 200, 220},
  };
  return palette[static_cast<std::size_t>(std::max(0, index)) % palette.size()];
}

bool init_nv12_tensor_meta(neat::Tensor& out, int width, int height, std::string& err) {
  if (width <= 0 || height <= 0 || (width % 2) != 0 || (height % 2) != 0) {
    err = "NV12 requires positive even width/height";
    return false;
  }
  out.dtype = neat::TensorDType::UInt8;
  out.layout = neat::TensorLayout::HW;
  out.shape = {height, width};
  out.strides_bytes = {width, 1};
  out.byte_offset = 0;
  out.device = {neat::DeviceType::CPU, 0};
  out.read_only = false;

  neat::ImageSpec image;
  image.format = neat::ImageSpec::PixelFormat::NV12;
  out.semantic.image = image;

  neat::Plane y;
  y.role = neat::PlaneRole::Y;
  y.shape = {height, width};
  y.strides_bytes = {width, 1};
  y.byte_offset = 0;

  neat::Plane uv;
  uv.role = neat::PlaneRole::UV;
  uv.shape = {height / 2, width};
  uv.strides_bytes = {width, 1};
  uv.byte_offset = static_cast<std::int64_t>(width) * static_cast<std::int64_t>(height);

  out.planes.clear();
  out.planes.push_back(std::move(y));
  out.planes.push_back(std::move(uv));
  return true;
}

std::uint8_t* nv12_y_plane(neat::Tensor& tensor) {
  if (!tensor.storage || !tensor.storage->data) {
    throw std::runtime_error("NV12 tensor has no CPU storage");
  }
  return static_cast<std::uint8_t*>(tensor.storage->data) + tensor.byte_offset;
}

std::uint8_t* nv12_uv_plane(neat::Tensor& tensor, int width, int height) {
  return nv12_y_plane(tensor) + static_cast<std::size_t>(width) * height;
}

void draw_point(neat::Tensor& frame, int width, int height, int x, int y, const Nv12Color& color) {
  if (x < 0 || y < 0 || x >= width || y >= height) {
    return;
  }
  auto* y_plane = nv12_y_plane(frame);
  auto* uv_plane = nv12_uv_plane(frame, width, height);
  y_plane[static_cast<std::size_t>(y) * width + x] = color.y;

  const int uv_x = x & ~1;
  const int uv_y = y / 2;
  if (uv_x + 1 < width && uv_y >= 0 && uv_y < height / 2) {
    const std::size_t offset = static_cast<std::size_t>(uv_y) * width + uv_x;
    uv_plane[offset] = color.u;
    uv_plane[offset + 1] = color.v;
  }
}

void fill_rect(neat::Tensor& frame, int width, int height, int x1, int y1, int x2, int y2,
               const Nv12Color& color) {
  x1 = std::clamp(x1, 0, width);
  x2 = std::clamp(x2, 0, width);
  y1 = std::clamp(y1, 0, height);
  y2 = std::clamp(y2, 0, height);
  if (x2 <= x1 || y2 <= y1) {
    return;
  }

  auto* y_plane = nv12_y_plane(frame);
  auto* uv_plane = nv12_uv_plane(frame, width, height);
  for (int y = y1; y < y2; ++y) {
    std::memset(y_plane + static_cast<std::size_t>(y) * width + x1, color.y,
                static_cast<std::size_t>(x2 - x1));
  }

  const int uv_x1 = x1 & ~1;
  const int uv_x2 = (x2 + 1) & ~1;
  const int uv_y1 = y1 / 2;
  const int uv_y2 = (y2 + 1) / 2;
  for (int row = std::max(0, uv_y1); row < std::min(height / 2, uv_y2); ++row) {
    auto* row_ptr = uv_plane + static_cast<std::size_t>(row) * width;
    for (int col = std::max(0, uv_x1); col + 1 < std::min(width, uv_x2); col += 2) {
      row_ptr[col] = color.u;
      row_ptr[col + 1] = color.v;
    }
  }
}

void draw_line(neat::Tensor& frame, int width, int height, int x1, int y1, int x2, int y2,
               int thickness, const Nv12Color& color) {
  const int dx = std::abs(x2 - x1);
  const int sx = x1 < x2 ? 1 : -1;
  const int dy = -std::abs(y2 - y1);
  const int sy = y1 < y2 ? 1 : -1;
  int err = dx + dy;
  while (true) {
    fill_rect(frame, width, height, x1 - thickness / 2, y1 - thickness / 2,
              x1 + thickness / 2 + 1, y1 + thickness / 2 + 1, color);
    if (x1 == x2 && y1 == y2) {
      break;
    }
    const int e2 = 2 * err;
    if (e2 >= dy) {
      err += dy;
      x1 += sx;
    }
    if (e2 <= dx) {
      err += dx;
      y1 += sy;
    }
  }
}

void draw_circle(neat::Tensor& frame, int width, int height, int cx, int cy, int radius,
                 const Nv12Color& color) {
  for (int y = -radius; y <= radius; ++y) {
    for (int x = -radius; x <= radius; ++x) {
      if (x * x + y * y <= radius * radius) {
        draw_point(frame, width, height, cx + x, cy + y, color);
      }
    }
  }
}

std::array<std::string_view, 7> glyph_for(char c) {
  switch (c) {
  case 'A': return {"01110", "10001", "10001", "11111", "10001", "10001", "10001"};
  case 'B': return {"11110", "10001", "10001", "11110", "10001", "10001", "11110"};
  case 'C': return {"01111", "10000", "10000", "10000", "10000", "10000", "01111"};
  case 'D': return {"11110", "10001", "10001", "10001", "10001", "10001", "11110"};
  case 'E': return {"11111", "10000", "10000", "11110", "10000", "10000", "11111"};
  case 'F': return {"11111", "10000", "10000", "11110", "10000", "10000", "10000"};
  case 'G': return {"01111", "10000", "10000", "10011", "10001", "10001", "01110"};
  case 'H': return {"10001", "10001", "10001", "11111", "10001", "10001", "10001"};
  case 'I': return {"11111", "00100", "00100", "00100", "00100", "00100", "11111"};
  case 'J': return {"00111", "00010", "00010", "00010", "10010", "10010", "01100"};
  case 'K': return {"10001", "10010", "10100", "11000", "10100", "10010", "10001"};
  case 'L': return {"10000", "10000", "10000", "10000", "10000", "10000", "11111"};
  case 'M': return {"10001", "11011", "10101", "10101", "10001", "10001", "10001"};
  case 'N': return {"10001", "11001", "10101", "10011", "10001", "10001", "10001"};
  case 'O': return {"01110", "10001", "10001", "10001", "10001", "10001", "01110"};
  case 'P': return {"11110", "10001", "10001", "11110", "10000", "10000", "10000"};
  case 'R': return {"11110", "10001", "10001", "11110", "10100", "10010", "10001"};
  case 'S': return {"01111", "10000", "10000", "01110", "00001", "00001", "11110"};
  case 'T': return {"11111", "00100", "00100", "00100", "00100", "00100", "00100"};
  case 'U': return {"10001", "10001", "10001", "10001", "10001", "10001", "01110"};
  case 'V': return {"10001", "10001", "10001", "10001", "10001", "01010", "00100"};
  case 'W': return {"10001", "10001", "10001", "10101", "10101", "10101", "01010"};
  case 'X': return {"10001", "10001", "01010", "00100", "01010", "10001", "10001"};
  case 'Y': return {"10001", "10001", "01010", "00100", "00100", "00100", "00100"};
  case 'Z': return {"11111", "00001", "00010", "00100", "01000", "10000", "11111"};
  case '0': return {"01110", "10001", "10011", "10101", "11001", "10001", "01110"};
  case '1': return {"00100", "01100", "00100", "00100", "00100", "00100", "01110"};
  case '2': return {"01110", "10001", "00001", "00010", "00100", "01000", "11111"};
  case '3': return {"11110", "00001", "00001", "01110", "00001", "00001", "11110"};
  case '4': return {"10010", "10010", "10010", "11111", "00010", "00010", "00010"};
  case '5': return {"11111", "10000", "10000", "11110", "00001", "00001", "11110"};
  case '6': return {"01111", "10000", "10000", "11110", "10001", "10001", "01110"};
  case '7': return {"11111", "00001", "00010", "00100", "01000", "01000", "01000"};
  case '8': return {"01110", "10001", "10001", "01110", "10001", "10001", "01110"};
  case '9': return {"01110", "10001", "10001", "01111", "00001", "00001", "11110"};
  case '-': return {"00000", "00000", "00000", "11111", "00000", "00000", "00000"};
  case '_': return {"00000", "00000", "00000", "00000", "00000", "00000", "11111"};
  case ':': return {"00000", "00100", "00100", "00000", "00100", "00100", "00000"};
  case '.': return {"00000", "00000", "00000", "00000", "00000", "01100", "01100"};
  case ' ': return {"00000", "00000", "00000", "00000", "00000", "00000", "00000"};
  default: return {"11111", "00001", "00010", "00100", "00100", "00000", "00100"};
  }
}

void draw_text(neat::Tensor& frame, int width, int height, int x, int y,
               const std::string& text, const Nv12Color& color, int scale = 2) {
  int cursor_x = x;
  for (char raw : text) {
    char c = raw >= 'a' && raw <= 'z' ? static_cast<char>(raw - 'a' + 'A') : raw;
    const auto glyph = glyph_for(c);
    for (int row = 0; row < static_cast<int>(glyph.size()); ++row) {
      for (int col = 0; col < static_cast<int>(glyph[row].size()); ++col) {
        if (glyph[row][col] == '1') {
          fill_rect(frame, width, height, cursor_x + col * scale, y + row * scale,
                    cursor_x + (col + 1) * scale, y + (row + 1) * scale, color);
        }
      }
    }
    cursor_x += 6 * scale;
    if (cursor_x >= width - 6 * scale) {
      break;
    }
  }
}

std::vector<float> tensor_to_float_vector(const neat::Tensor& tensor) {
  if (tensor.dtype != neat::TensorDType::Float32) {
    return {};
  }
  const auto bytes = tensor.copy_dense_bytes_tight();
  if (bytes.empty() || (bytes.size() % sizeof(float)) != 0) {
    return {};
  }
  std::vector<float> values(bytes.size() / sizeof(float));
  std::memcpy(values.data(), bytes.data(), values.size() * sizeof(float));
  return values;
}

struct PoseKeypoint {
  float x = 0.0f;
  float y = 0.0f;
  float score = 0.0f;
  int id = -1;
  int type = -1;
};

struct PosePerson {
  std::array<int, 18> keypoint_ids{};
  float score = 0.0f;
  int count = 0;
  PosePerson() { keypoint_ids.fill(-1); }
};

struct OpenPoseTensorData {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::vector<float> values;
};

bool openpose_tensor_geometry(const neat::Tensor& tensor, const std::vector<float>& values,
                              int& heat_w, int& heat_h, int& channels) {
  if (tensor.shape.size() >= 3) {
    heat_h = static_cast<int>(tensor.shape[tensor.shape.size() - 3]);
    heat_w = static_cast<int>(tensor.shape[tensor.shape.size() - 2]);
    channels = static_cast<int>(tensor.shape.back());
  }
  return heat_w > 0 && heat_h > 0 && channels >= 18 &&
         values.size() >= static_cast<std::size_t>(heat_w) * heat_h * channels;
}

float tensor_channel_value(const OpenPoseTensorData& tensor, int x, int y, int channel) {
  x = std::clamp(x, 0, tensor.width - 1);
  y = std::clamp(y, 0, tensor.height - 1);
  channel = std::clamp(channel, 0, tensor.channels - 1);
  const std::size_t offset =
      (static_cast<std::size_t>(y) * tensor.width + x) * tensor.channels + channel;
  return offset < tensor.values.size() ? tensor.values[offset] : 0.0f;
}

std::pair<int, int> model_to_frame(float model_x, float model_y, int frame_w, int frame_h) {
  constexpr int kModelSize = 480;
  const double scale =
      std::min(static_cast<double>(kModelSize) / std::max(1, frame_w),
               static_cast<double>(kModelSize) / std::max(1, frame_h));
  const double pad_x = (kModelSize - frame_w * scale) * 0.5;
  const double pad_y = (kModelSize - frame_h * scale) * 0.5;
  const int frame_x = static_cast<int>(std::lround((model_x - pad_x) / std::max(1e-9, scale)));
  const int frame_y = static_cast<int>(std::lround((model_y - pad_y) / std::max(1e-9, scale)));
  return {std::clamp(frame_x, 0, frame_w - 1), std::clamp(frame_y, 0, frame_h - 1)};
}

std::vector<std::vector<PoseKeypoint>> find_openpose_peaks(const OpenPoseTensorData& heatmap) {
  constexpr float kPeakThreshold = 0.15f;
  constexpr float kSuppressRadius = 6.0f;
  constexpr int kMaxPeaksPerType = 16;
  std::vector<std::vector<PoseKeypoint>> by_type(18);
  int next_id = 0;

  for (int type = 0; type < 18; ++type) {
    std::vector<PoseKeypoint> candidates;
    for (int y = 1; y + 1 < heatmap.height; ++y) {
      for (int x = 1; x + 1 < heatmap.width; ++x) {
        const float score = tensor_channel_value(heatmap, x, y, type);
        if (score < kPeakThreshold || score <= tensor_channel_value(heatmap, x - 1, y, type) ||
            score <= tensor_channel_value(heatmap, x + 1, y, type) ||
            score <= tensor_channel_value(heatmap, x, y - 1, type) ||
            score <= tensor_channel_value(heatmap, x, y + 1, type)) {
          continue;
        }
        candidates.push_back({static_cast<float>(x), static_cast<float>(y), score, -1, type});
      }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const PoseKeypoint& a, const PoseKeypoint& b) { return a.score > b.score; });
    for (const auto& candidate : candidates) {
      bool suppressed = false;
      for (const auto& kept : by_type[static_cast<std::size_t>(type)]) {
        const float dx = candidate.x - kept.x;
        const float dy = candidate.y - kept.y;
        if (std::sqrt(dx * dx + dy * dy) < kSuppressRadius) {
          suppressed = true;
          break;
        }
      }
      if (!suppressed) {
        PoseKeypoint kept = candidate;
        kept.id = next_id++;
        by_type[static_cast<std::size_t>(type)].push_back(kept);
      }
      if (static_cast<int>(by_type[static_cast<std::size_t>(type)].size()) >= kMaxPeaksPerType) {
        break;
      }
    }
  }
  return by_type;
}

struct PoseConnection {
  int a_id = -1;
  int b_id = -1;
  float score = 0.0f;
  int a_local = -1;
  int b_local = -1;
};

std::optional<float> paf_connection_score(const OpenPoseTensorData& paf, const PoseKeypoint& a,
                                          const PoseKeypoint& b, int paf_x_channel,
                                          int paf_y_channel) {
  constexpr int kSamples = 10;
  constexpr float kMinPafScore = 0.08f;
  constexpr float kMinSuccessRatio = 0.80f;
  const float dx = b.x - a.x;
  const float dy = b.y - a.y;
  const float norm = std::sqrt(dx * dx + dy * dy);
  if (norm < 1.0f) {
    return std::nullopt;
  }
  const float ux = dx / norm;
  const float uy = dy / norm;
  float score_sum = 0.0f;
  int valid = 0;
  for (int i = 0; i < kSamples; ++i) {
    const float t = static_cast<float>(i) / (kSamples - 1);
    const int x = static_cast<int>(std::lround(a.x + dx * t));
    const int y = static_cast<int>(std::lround(a.y + dy * t));
    const float dot = tensor_channel_value(paf, x, y, paf_x_channel) * ux +
                      tensor_channel_value(paf, x, y, paf_y_channel) * uy;
    if (dot > kMinPafScore) {
      score_sum += dot;
      ++valid;
    }
  }
  if (static_cast<float>(valid) / kSamples < kMinSuccessRatio || valid == 0) {
    return std::nullopt;
  }
  return score_sum / valid;
}

std::vector<PoseConnection> match_limb_connections(
    const std::vector<PoseKeypoint>& a_candidates,
    const std::vector<PoseKeypoint>& b_candidates, const OpenPoseTensorData& paf,
    int paf_x_channel, int paf_y_channel) {
  std::vector<PoseConnection> candidates;
  for (std::size_t ai = 0; ai < a_candidates.size(); ++ai) {
    for (std::size_t bi = 0; bi < b_candidates.size(); ++bi) {
      const auto score =
          paf_connection_score(paf, a_candidates[ai], b_candidates[bi], paf_x_channel,
                               paf_y_channel);
      if (score.has_value()) {
        candidates.push_back({a_candidates[ai].id, b_candidates[bi].id, *score,
                              static_cast<int>(ai), static_cast<int>(bi)});
      }
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const PoseConnection& a, const PoseConnection& b) { return a.score > b.score; });

  std::vector<PoseConnection> out;
  std::vector<bool> used_a(a_candidates.size(), false);
  std::vector<bool> used_b(b_candidates.size(), false);
  for (const auto& candidate : candidates) {
    if (used_a[static_cast<std::size_t>(candidate.a_local)] ||
        used_b[static_cast<std::size_t>(candidate.b_local)]) {
      continue;
    }
    used_a[static_cast<std::size_t>(candidate.a_local)] = true;
    used_b[static_cast<std::size_t>(candidate.b_local)] = true;
    out.push_back(candidate);
  }
  return out;
}

void add_pose_connection(PosePerson& person, int ka, int kb, const PoseConnection& connection,
                         const std::vector<PoseKeypoint>& all_keypoints) {
  if (person.keypoint_ids[static_cast<std::size_t>(ka)] < 0) {
    person.keypoint_ids[static_cast<std::size_t>(ka)] = connection.a_id;
    person.count += 1;
    person.score += all_keypoints[static_cast<std::size_t>(connection.a_id)].score;
  }
  if (person.keypoint_ids[static_cast<std::size_t>(kb)] < 0) {
    person.keypoint_ids[static_cast<std::size_t>(kb)] = connection.b_id;
    person.count += 1;
    person.score += all_keypoints[static_cast<std::size_t>(connection.b_id)].score;
  }
  person.score += connection.score;
}

void merge_pose_people(PosePerson& target, const PosePerson& source) {
  for (std::size_t i = 0; i < target.keypoint_ids.size(); ++i) {
    if (target.keypoint_ids[i] < 0 && source.keypoint_ids[i] >= 0) {
      target.keypoint_ids[i] = source.keypoint_ids[i];
    }
  }
  target.score += source.score;
  target.count = 0;
  for (int id : target.keypoint_ids) {
    if (id >= 0) {
      ++target.count;
    }
  }
}

std::vector<PosePerson> assemble_openpose_people(
    const std::vector<std::vector<PoseKeypoint>>& by_type,
    const std::vector<PoseKeypoint>& all_keypoints, const OpenPoseTensorData& paf) {
  static constexpr std::array<std::pair<int, int>, 19> kBodyParts = {
      std::pair{1, 2},  {1, 5},  {2, 3},   {3, 4},   {5, 6},   {6, 7},  {1, 8},
      {8, 9},          {9, 10}, {1, 11},  {11, 12}, {12, 13}, {1, 0},  {0, 14},
      {14, 16},        {0, 15}, {15, 17}, {2, 16},  {5, 17},
  };
  static constexpr std::array<std::pair<int, int>, 19> kPafParts = {
      std::pair{12, 13}, {20, 21}, {14, 15}, {16, 17}, {22, 23}, {24, 25}, {0, 1},
      {2, 3},           {4, 5},   {6, 7},   {8, 9},   {10, 11}, {28, 29}, {30, 31},
      {34, 35},         {32, 33}, {36, 37}, {18, 19}, {26, 27},
  };

  std::vector<PosePerson> people;
  for (std::size_t part_id = 0; part_id < kBodyParts.size(); ++part_id) {
    const auto [ka, kb] = kBodyParts[part_id];
    const auto& a_candidates = by_type[static_cast<std::size_t>(ka)];
    const auto& b_candidates = by_type[static_cast<std::size_t>(kb)];
    if (a_candidates.empty() || b_candidates.empty()) {
      continue;
    }
    const auto [paf_x, paf_y] = kPafParts[part_id];
    const auto connections = match_limb_connections(a_candidates, b_candidates, paf, paf_x, paf_y);

    for (const auto& connection : connections) {
      std::vector<int> matched_indices;
      for (std::size_t i = 0; i < people.size(); ++i) {
        if (people[i].keypoint_ids[static_cast<std::size_t>(ka)] == connection.a_id ||
            people[i].keypoint_ids[static_cast<std::size_t>(kb)] == connection.b_id) {
          matched_indices.push_back(static_cast<int>(i));
        }
      }
      if (!matched_indices.empty()) {
        auto& person = people[static_cast<std::size_t>(matched_indices.front())];
        add_pose_connection(person, ka, kb, connection, all_keypoints);
        for (std::size_t idx = matched_indices.size(); idx-- > 1;) {
          const int source_index = matched_indices[idx];
          merge_pose_people(person, people[static_cast<std::size_t>(source_index)]);
          people.erase(people.begin() + source_index);
        }
      } else {
        PosePerson person;
        person.keypoint_ids[static_cast<std::size_t>(ka)] = connection.a_id;
        person.keypoint_ids[static_cast<std::size_t>(kb)] = connection.b_id;
        person.count = 2;
        person.score = all_keypoints[static_cast<std::size_t>(connection.a_id)].score +
                       all_keypoints[static_cast<std::size_t>(connection.b_id)].score +
                       connection.score;
        people.push_back(person);
      }
    }
  }

  std::vector<PosePerson> filtered;
  for (const auto& person : people) {
    if (person.count >= 4 && person.score / std::max(1, person.count) >= 0.35f) {
      filtered.push_back(person);
    }
  }
  return filtered;
}

} // namespace

bool infer_nv12_dims(const neat::Tensor& tensor, int& width, int& height) {
  width = tensor.width();
  height = tensor.height();
  if ((width <= 0 || height <= 0) && tensor.shape.size() >= 2) {
    height = static_cast<int>(tensor.shape[0]);
    width = static_cast<int>(tensor.shape[1]);
  }
  return width > 0 && height > 0;
}

neat::Tensor make_blank_nv12_tensor(int width, int height) {
  std::string err;
  const std::size_t bytes =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U / 2U;
  neat::Tensor out;
  out.storage = neat::make_cpu_owned_storage(bytes);
  std::memset(out.storage->data, 0, bytes);
  if (!init_nv12_tensor_meta(out, width, height, err)) {
    throw std::runtime_error(err);
  }
  return out;
}

neat::Tensor copy_nv12_to_cpu_tensor(const neat::Tensor& input) {
  int width = 0;
  int height = 0;
  if (!input.is_nv12() || !infer_nv12_dims(input, width, height)) {
    throw std::runtime_error("invalid NV12 tensor");
  }

  const auto bytes = input.copy_nv12_contiguous();
  if (bytes.empty()) {
    throw std::runtime_error("NV12 copy failed");
  }
  neat::Tensor out;
  out.storage = neat::make_cpu_owned_storage(bytes.size());
  std::memcpy(out.storage->data, bytes.data(), bytes.size());
  std::string err;
  if (!init_nv12_tensor_meta(out, width, height, err)) {
    throw std::runtime_error(err);
  }
  return out;
}

PoseOverlayStats draw_openpose_overlay(neat::Tensor& frame, const neat::Sample& model_sample) {
  PoseOverlayStats stats;
  int width = 0;
  int height = 0;
  if (!infer_nv12_dims(frame, width, height)) {
    return stats;
  }

  const Nv12Color white{235, 128, 128};
  const Nv12Color black{16, 128, 128};
  const Nv12Color header{160, 210, 190};
  fill_rect(frame, width, height, 0, 0, width, 34, black);
  draw_text(frame, width, height, 10, 8, "open_pose", header, 2);
  draw_text(frame, width, height, 220, 8, sample_summary(model_sample), white, 2);

  const auto tensors = neat::tensors_from_sample(model_sample, false);
  OpenPoseTensorData heatmap;
  OpenPoseTensorData paf;
  for (const auto& tensor : tensors) {
    auto values = tensor_to_float_vector(tensor);
    int candidate_w = 0;
    int candidate_h = 0;
    int candidate_c = 0;
    if (!openpose_tensor_geometry(tensor, values, candidate_w, candidate_h, candidate_c)) {
      continue;
    }
    if (candidate_c == 19) {
      heatmap = {candidate_w, candidate_h, candidate_c, std::move(values)};
    } else if (candidate_c == 38) {
      paf = {candidate_w, candidate_h, candidate_c, std::move(values)};
    }
  }
  if (heatmap.values.empty() || paf.values.empty() || heatmap.width != paf.width ||
      heatmap.height != paf.height) {
    draw_text(frame, width, height, 520, 8, "PERSONS:0 KPTS:0", white, 2);
    fill_rect(frame, width, height, 0, 34, width, 37, header);
    return stats;
  }

  const auto by_type = find_openpose_peaks(heatmap);
  std::vector<PoseKeypoint> all_keypoints;
  for (const auto& candidates : by_type) {
    for (const auto& candidate : candidates) {
      if (candidate.id >= static_cast<int>(all_keypoints.size())) {
        all_keypoints.resize(static_cast<std::size_t>(candidate.id) + 1);
      }
      all_keypoints[static_cast<std::size_t>(candidate.id)] = candidate;
    }
  }

  const auto people = assemble_openpose_people(by_type, all_keypoints, paf);
  stats.persons = static_cast<int>(people.size());

  static constexpr std::array<std::pair<int, int>, 17> kDrawSkeleton = {
      std::pair{1, 2},  {1, 5},  {2, 3},   {3, 4},   {5, 6},   {6, 7},
      {1, 8},          {8, 9},  {9, 10},  {1, 11},  {11, 12}, {12, 13},
      {1, 0},          {0, 14}, {14, 16}, {0, 15},  {15, 17},
  };
  for (std::size_t person_index = 0; person_index < people.size(); ++person_index) {
    const Nv12Color person_color = pose_color(static_cast<int>(person_index) + 40);
    const auto& person = people[person_index];
    for (const auto& [a, b] : kDrawSkeleton) {
      const int id_a = person.keypoint_ids[static_cast<std::size_t>(a)];
      const int id_b = person.keypoint_ids[static_cast<std::size_t>(b)];
      if (id_a < 0 || id_b < 0 || id_a >= static_cast<int>(all_keypoints.size()) ||
          id_b >= static_cast<int>(all_keypoints.size())) {
        continue;
      }
      const auto& ka = all_keypoints[static_cast<std::size_t>(id_a)];
      const auto& kb = all_keypoints[static_cast<std::size_t>(id_b)];
      const auto [fax, fay] =
          model_to_frame((ka.x + 0.5f) * 480.0f / heatmap.width,
                         (ka.y + 0.5f) * 480.0f / heatmap.height, width, height);
      const auto [fbx, fby] =
          model_to_frame((kb.x + 0.5f) * 480.0f / heatmap.width,
                         (kb.y + 0.5f) * 480.0f / heatmap.height, width, height);
      draw_line(frame, width, height, fax, fay, fbx, fby, 4, person_color);
    }
    for (int id : person.keypoint_ids) {
      if (id < 0 || id >= static_cast<int>(all_keypoints.size())) {
        continue;
      }
      const auto& kp = all_keypoints[static_cast<std::size_t>(id)];
      const auto [fx, fy] =
          model_to_frame((kp.x + 0.5f) * 480.0f / heatmap.width,
                         (kp.y + 0.5f) * 480.0f / heatmap.height, width, height);
      draw_circle(frame, width, height, fx, fy, 5, white);
    }
    stats.keypoints += person.count;
  }

  draw_text(frame, width, height, 520, 8,
            "PERSONS:" + std::to_string(stats.persons) + " KPTS:" +
                std::to_string(stats.keypoints),
            white, 2);
  fill_rect(frame, width, height, 0, 34, width, 37, header);
  return stats;
}

std::string sample_summary(const neat::Sample& sample) {
  const auto tensors = neat::tensors_from_sample(sample, false);
  return std::to_string(tensors.size()) + " tensors";
}

std::string tensor_shape_summary(const neat::Sample& sample) {
  const auto tensors = neat::tensors_from_sample(sample, false);
  std::string out;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    if (!out.empty()) {
      out += " ";
    }
    out += "t" + std::to_string(i) + "=[";
    for (std::size_t j = 0; j < tensors[i].shape.size(); ++j) {
      if (j > 0) {
        out += "x";
      }
      out += std::to_string(tensors[i].shape[j]);
    }
    out += "]";
  }
  return out;
}
