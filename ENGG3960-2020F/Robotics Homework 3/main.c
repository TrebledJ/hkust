#include "qdbmp.h"
#include <stdio.h>
#include <stdint.h>
#include <math.h>


void median_filter(BMP* imageIn, BMP* imageOut)
{
    printf("Applying median filter.\n");

    int width  = BMP_GetWidth(imageIn);
    int height = BMP_GetHeight(imageIn);

    uint8_t window[9];
    for (size_t y=1; y<height-1; y++)
        for (size_t x=1; x<width-1; x++)
        {
            // Fill the 3x3 window
            for (int u=x-1, k=0; u<=x+1; u++)
                for (int v=y-1; v<=y+1; v++)
                    window[k++] = BMP_GetPixelGray(imageIn, u, v);

            // Find the median
            for(uint8_t i=0; i<5; i++)
                for (uint8_t j=i+1; j<9; j++)
                    if (window[j] < window[i])
                    {
                        uint8_t temp = window[i];
                        window[i] = window[j];
                        window[j] = temp;
                    }

            BMP_SetPixelGray(imageOut, x, y, window[4]);
        }
}


double matmul33_combine(double* a, double* b)
{
    double sum = 0;
    for (int row = 0; row < 3; row++)
        for (int col = 0; col < 3; col++)
        {
            int i = row * 3 + col;
            sum += a[i] * b[i];
        }
    return sum;
}


void sobel_filter(BMP* imageIn, BMP* imageOut)
{
    printf("Applying sobel filter.\n");

    int width  = BMP_GetWidth(imageIn);
    int height = BMP_GetHeight(imageIn);

    double gx_mat[9] = {
        -1, 0, +1,
        -2, 0, +2,
        -1, 0, +1,
    };
    double gy_mat[9] = {
        -1, -2, -1,
        0, 0, 0,
        +1, +2, +1,
    };

    for (size_t y=1; y<height-1; y++)
        for (size_t x=1; x<width-1; x++)
        {
            double window[9];
            // Fill the 3x3 window
            for (int u=x-1, k=0; u<=x+1; u++)
                for (int v=y-1; v<=y+1; v++)
                    window[k++] = BMP_GetPixelGray(imageIn, u, v);

            double gx = matmul33_combine(window, gx_mat);
            double gy = matmul33_combine(window, gy_mat);

            double mag = sqrt(gx*gx + gy*gy) / 2;
            BMP_SetPixelGray(imageOut, x, y, (int)mag);
        }
}


int main(void)
{
    BMP* imageIn = BMP_ReadFile("lenna.bmp");
    BMP_CHECK_ERROR(stdout, -1);

    int width  = BMP_GetWidth(imageIn);
    int height = BMP_GetHeight(imageIn);

    BMP* imageOut = BMP_Create(width, height, 8);
    for (int i=0; i<256; i++)
    	BMP_SetPaletteColor(imageOut, i, i, i, i);

    // median_filter(imageIn, imageOut);
    sobel_filter(imageIn, imageOut);

    BMP_WriteFile(imageOut, "output.bmp");
    BMP_Free(imageIn);
    BMP_Free(imageOut);
    return 0;
}
