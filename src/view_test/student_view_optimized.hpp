#ifndef STUDENT_VIEW_OPTIMIZED_HPP_
#define STUDENT_VIEW_OPTIMIZED_HPP_
#include "../../utils/ppm.hpp"
#include "../../utils/commonCUDA.hpp"
#include "../../utils/chronoGPU.hpp"
#include "../../utils/variable.hpp"
#include "../../utils/geometry.hpp"

__device__ double angle_calc_optimized( float xa, float ya, uint8_t za, float xb, float yb, uint8_t zb );
__global__ void kernel_optimized_view_test( const uint8_t *in, uint8_t *out, uint32_t width, uint32_t height, int32_t cx, int32_t cy, uint8_t cz );

float view_test_GPU_optimized(los::Heightmap in, los::Heightmap &out , Point center);

#endif