#ifndef PROJETGPU_STUDENT_VIEW_DOUBLE_KERNEL_HPP
#define PROJETGPU_STUDENT_VIEW_DOUBLE_KERNEL_HPP

#include "../../utils/ppm.hpp"
#include "../../utils/commonCUDA.hpp"
#include "../../utils/chronoGPU.hpp"
#include "../../utils/variable.hpp"
#include "../../utils/geometry.hpp"


__global__ void kernel_angle_calc(const uint8_t *in, uint32_t width, uint32_t height, int32_t cx, int32_t cy, uint8_t cz, float *angles_tab);
__global__ void kernel_view_test_tab(const uint8_t *out, uint32_t width, uint32_t height, int32_t cx, int32_t cy, float *angles_tab);

__device__ float angle_calc_double_kernel(float xa,float ya,uint8_t za,float xb,float yb,uint8_t zb);

float view_test_double_kernel(los::Heightmap in, los::Heightmap &out , Point center);

#endif //PROJETGPU_STUDENT_VIEW_DOUBLE_KERNEL_HPP
