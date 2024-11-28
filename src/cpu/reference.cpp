#include "reference.hpp"

/*
Fonction de visibilite sur le CPU
On regarde pour chaque point si tous les points entre lui et le centre ont un angle plus petit
Si oui alors ce point est visible depuis le centre
*/
float view_test_CPU(const los::Heightmap &in, los::Heightmap &out, Point c)
{
    const uint32_t width = in.getWidth();
    const uint32_t height = in.getHeight();
    double *angle_calc = new double[width * height];

    ChronoCPU chr;
    chr.start();
    /*

        Calcul des angles entre chaque point et le centre
        On stocke ensuite tous ces points dans un tableau
        pour pouvoir les reutiliser plus tard

    */
    for (int m = 0; m < height; m++)
    {
        for (int l = 0; l < width; l++)
        {
            angle_calc[l + m * width] = Point::angle(c, Point(l, m, in.getPixel(l, m)));
        }
    }

    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {
            // Si le point est le meme que le centre , on passe au prochain et on le colorie en rouge
            if (i == c.getX() && j == c.getY())
            {
                out.setPixel(i, j, 155);
                continue;
            }
            // Rasterisation de la droite
            uint32_t D = max(abs(i - c.getX()), abs(j - c.getY()));
            float stepx = (i - c.getX()) / (float)D;
            float stepy = (j - c.getY()) / (float)D;

            // Construction des D cases entre le point et le centre
            uint32_t k = 1;
            bool isVisible = true;
            float arctanPoint = angle_calc[i + j * width];
            while (k < D && isVisible)
            {
                int xi = c.getX() + stepx * k;
                int yj = c.getY() + stepy * k;

                uint8_t z = in.getPixel(xi, yj);
                /*
                Verification de la visibilite
                Si un point a un angle plus grand que le point courant
                Alors notre point n'est pas visible depuis le centre
                */
                if (angle_calc[xi + yj * width] >= arctanPoint)
                {
                    isVisible = false;
                }
                k++;
            }

            if (isVisible)
            {
                out.setPixel(i, j, 255);
            }
            else
            {
                out.setPixel(i, j, 0);
            }
        }
    }
    chr.stop();
    return chr.elapsedTime();
}

// Reduction d'image
float tiled_CPU(const los::Heightmap &in, los::Heightmap &out)
{
    const uint32_t inWidth = in.getWidth();
    const uint32_t inHeight = in.getHeight();
    const uint32_t outWidth = out.getWidth();
    const uint32_t outHeight = out.getHeight();

    // Instanciation de l'image de sortie
    for (int i = 0; i < outWidth; i++)
    {
        for (int j = 0; j < outHeight; j++)
        {
            out.setPixel(i, j, 0);
        }
    }

    ChronoCPU chr;
    chr.start();
    for (int i = 0; i < inWidth; i++)
    {
        for (int j = 0; j < inHeight; j++)
        {
            /*
                On regarde a quel pixel de l'image de sortie
                correspond le pixel d'entree
            */
            int32_t index_x = i / ceilf(inWidth / outWidth);
            int32_t index_y = j / ceilf(inHeight / outHeight);
            if (index_x < outWidth && index_y < outHeight)
            {
                uint8_t pixelValue = out.getPixel(index_x, index_y);
                // Maximum entre le pixel d'entree de l'image et la valeur presente dans la sortie
                out.setPixel(index_x, index_y, max(pixelValue, in.getPixel(i, j)));
            }
        }
    }
    chr.stop();
    return chr.elapsedTime();
}