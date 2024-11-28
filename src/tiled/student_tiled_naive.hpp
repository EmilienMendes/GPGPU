#ifndef STUDENT_TILED_HPP_
#define STUDENT_TILED_HPP_
#include "../../utils/ppm.hpp"
#include "../../utils/commonCUDA.hpp"
#include "../../utils/chronoGPU.hpp"
#include "../../utils/variable.hpp"
#include "../../utils/geometry.hpp"

__global__ void kernel_naive_tiled(const uint8_t *dev_inPtr, uint32_t *dev_outPtr, uint32_t inWidth,
                                   uint32_t inHeight,uint32_t outWidth, uint32_t outHeight);
float tiled_GPU_naive(los::Heightmap in, los::Heightmap &out);

#endif