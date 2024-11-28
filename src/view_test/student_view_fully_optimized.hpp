#ifndef PROJETGPU_STUDENT_VIEW_FULLY_OPTIMIZED_HPP
#define PROJETGPU_STUDENT_VIEW_FULLY_OPTIMIZED_HPP

#include "../../utils/ppm.hpp"
#include "../../utils/commonCUDA.hpp"
#include "../../utils/chronoGPU.hpp"
#include "../../utils/variable.hpp"
#include "../../utils/geometry.hpp"

__global__ void kernel_fully_optimized_view_test( const uint8_t *in, uint8_t *out, uint32_t width, uint32_t height,int32_t cx,int32_t cy,uint8_t cz  );
__device__ float angle_calc_fully_optimized(float xa,float ya,uint8_t za,float xb,float yb,uint8_t zb);

float view_test_GPU_fully_optimized(los::Heightmap in, los::Heightmap &out , Point center);

#endif //PROJETGPU_STUDENT_VIEW_FULLY_OPTIMIZED_HPP
