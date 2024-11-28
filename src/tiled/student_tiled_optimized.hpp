#ifndef STUDENT_TILED_OPTIMIZED_HPP_
#define STUDENT_TILED_OPTIMIZED_HPP_
#include "../../utils/ppm.hpp"
#include "../../utils/commonCUDA.hpp"
#include "../../utils/chronoGPU.hpp"
#include "../../utils/variable.hpp"
#include "../../utils/geometry.hpp"


__global__ void  kernel_optimized_tiled( const uint8_t *in, uint32_t *out,uint32_t inWidth,uint32_t inHeight,uint32_t outWidth,uint32_t outHeight );

float tiled_GPU_optimized(los::Heightmap in, los::Heightmap &out);

#endif