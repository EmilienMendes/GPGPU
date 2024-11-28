#include "ppm.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
namespace los
{

    Heightmap::Heightmap( const uint32_t width, const uint32_t height ) : m_width( width ), m_height( height )
    {
        m_pixels = std::vector<uint8_t>( m_width * m_height, 0u );
    }

    Heightmap::Heightmap( const char * const name ) : m_width( 0 ), m_height( 0 ), m_pixels()
    {
        std::ifstream file( name, std::ios::in | std::ios::binary );
        if ( !file )
        {
            std::cerr << "Cannot open PPM file " << name << std::endl;
            exit( EXIT_FAILURE );
        }
        std::string MagicNumber, line;

        // MagicNumber
        getLine( file, line );
        std::istringstream iss1( line );
        iss1 >> MagicNumber;
        if ( MagicNumber != "P6" )
        { // Binary ? or nothing ?
            std::cerr << "Error reading PPM file " << name << ": unknown Magic Number \"" << MagicNumber
                      << "\". Only P6 is supported" << std::endl
                      << std::endl;
            exit( EXIT_FAILURE );
        }

        // Image size
        getLine( file, line );
        std::istringstream iss2( line );

        int width, height;
        iss2 >> width >> height;
        if ( width <= 0 || height <= 0 )
        {
            std::cerr << "Wrong image size " << width << " x " << height << std::endl;
            exit( EXIT_FAILURE );
        }

        m_width  = width;
        m_height = height;

        // Max channel value
        int maxChannelVal;
        getLine( file, line );
        std::istringstream iss3( line );
        iss3 >> maxChannelVal;
        if ( maxChannelVal > 255 )
        {
            std::cerr << "Max channel value too high in " << name << std::endl;
            exit( EXIT_FAILURE );
        }

        // Read pixels
        std::vector<uint8_t> buffer = std::vector<uint8_t>( m_width * m_height * 3 );
        for ( uint32_t y = m_height; y-- > 0; )
        {
            // Reading each line
            file.read( reinterpret_cast<char *>( buffer.data() + ( y * m_width ) * 3 ), m_width * 3 );
        }

        const std::size_t pixelNb = m_width * m_height;
        m_pixels                  = std::vector<uint8_t>( pixelNb );

        // Copy only red channel
        for ( std::size_t i = 0; i < pixelNb; i++ )
            m_pixels[ i ] = buffer[ i * 3 ];
    }

    void Heightmap::getLine( std::ifstream & file, std::string & s ) const
    {
        for ( ;; )
        {
            if ( !std::getline( file, s ) )
            {
                std::cerr << "Error reading PPM file" << std::endl;
                exit( EXIT_FAILURE );
            }
            std::string::size_type index = s.find_first_not_of( "\n\r\t " );
            if ( index != std::string::npos && s[ index ] != '#' )
                break;
        }
    }

    void Heightmap::saveTo( const char * name ) const
    {
        std::ofstream file( name, std::ios::out | std::ios::trunc | std::ios::binary );
        file << "P6" << std::endl;                       // Magic Number !
        file << m_width << " " << m_height << std::endl; // Image size
        file << "255" << std::endl;                      // Max R G B

        const std::size_t    pixelNb = m_width * m_height;
        std::vector<uint8_t> buffer  = std::vector<uint8_t>( pixelNb * 3 );
        for ( std::size_t i = 0; i < pixelNb; i++ )
        {
            const uint8_t height = m_pixels[ i ];
            buffer[ i * 3 + 0 ]  = height;
            buffer[ i * 3 + 1 ]  = height;
            buffer[ i * 3 + 2 ]  = height;

            if(height == 155) {
                buffer[ i * 3 + 0 ]  = 255;
                buffer[ i * 3 + 1 ]  = 0;
                buffer[ i * 3 + 2 ]  = 0;
            }
        }

        const uint8_t * const ptr = buffer.data();
        for ( int y = m_height; y-- > 0; )
        {
            // Writing each line
            file.write( (char *)( ptr + y * m_width * 3 ), m_width * 3 );
        }
        file.close();
    }
} // namespace los