#ifndef REFERENCE_HPP_
#define REFERENCE_HPP_
#include "../../utils/ppm.hpp"
#include "../../utils/chronoCPU.hpp"
#include "../../utils/geometry.hpp"

float view_test_CPU( const los::Heightmap &in, los::Heightmap &out,Point center);

float tiled_CPU( const los::Heightmap &in, los::Heightmap &out);
#endif // REFERENCE_HPP_