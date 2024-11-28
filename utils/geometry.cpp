#include "geometry.hpp"

Point::Point(int32_t p_x, int32_t p_y, uint8_t p_z) : x(p_x), y(p_y), z(p_z) {}

void Point::PrintPoint() const {
    std::cout << "{" << x << "," << y << "," << (int)z << "}" << std::endl;
}

float Point::angle(Point a,Point b){
        float denominator = sqrtf( ( (float) b.getX() - (float)a.getX() ) * ( (float) b.getX() - (float)a.getX() ) + 
        ( (float) b.getY() - (float)a.getY() ) * ( (float) b.getY() - (float)a.getY() ) );
        return atan(( (float) b.getZ() - (float) a.getZ() )/denominator);
}


