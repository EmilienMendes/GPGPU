#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

#include <string>
#include <iosfwd>
#include <iostream>

class Point
{
public:
    Point() = delete; // undefined
    Point( int32_t x, int32_t y,uint8_t z );
    ~Point() = default;

    void PrintPoint() const;
    static float angle(Point a, Point b);

    int32_t   getX() const  { return x; }
    int32_t   getY() const { return y; }
    uint8_t   getZ() const { return z; }

private:
    int32_t x;
    int32_t y;
    uint8_t z;

};


#endif
