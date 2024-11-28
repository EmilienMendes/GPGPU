#ifndef __PP_HPP__
#define __PP_HPP__

#include <vector>
#include <string>
#include <iosfwd>

namespace los
{
    class Heightmap
    {
    public:
        Heightmap() = delete; // undefined
        Heightmap( const uint32_t width, const uint32_t height );
        Heightmap( const char * const name );
        ~Heightmap() = default;

        uint32_t    getWidth() const { return m_width; }
        uint32_t    getHeight() const { return m_height; }
        std::size_t getSize() const { return std::size_t( m_width ) * std::size_t( m_height ); }
        uint8_t *   getPtr() { return m_pixels.data(); }

        uint8_t getPixel( const unsigned x, const unsigned y ) const { return m_pixels[ x + y * m_width ]; }
        void    setPixel( const unsigned x, const unsigned y, const uint8_t height )
        {
            m_pixels[ x + y * m_width ] = height;
        }

        void getLine( std::ifstream & file, std::string & s ) const;
        void saveTo( const char * name ) const;

    private:
        uint32_t             m_width;
        uint32_t             m_height;
        std::vector<uint8_t> m_pixels;
    };
} // namespace los

#endif // __PP_HPP__