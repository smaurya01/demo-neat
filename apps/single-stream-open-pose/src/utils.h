#pragma once

#include <neat.h>

#include <string>

struct PoseOverlayStats {
  int persons = 0;
  int keypoints = 0;
};

bool infer_nv12_dims(const simaai::neat::Tensor& tensor, int& width, int& height);
simaai::neat::Tensor make_blank_nv12_tensor(int width, int height);
simaai::neat::Tensor copy_nv12_to_cpu_tensor(const simaai::neat::Tensor& input);

PoseOverlayStats draw_openpose_overlay(simaai::neat::Tensor& frame,
                                       const simaai::neat::Sample& model_sample);

std::string sample_summary(const simaai::neat::Sample& sample);
std::string tensor_shape_summary(const simaai::neat::Sample& sample);
