#include "kernels_2d.h"
#include <math.h>

static inline int clamp(int val, int max_val) {
    if (val < 0) return 0;
    if (val >= max_val) return max_val - 1;
    return val;
}

/* Reference Gaussian 3x3 (17 flops/pixel, clamp borders) */
void gauss3_v0_serial(const float* in, float* out, int width, int height, const TuningConfig* cfg) {
    (void)cfg;
    const float k[3][3] = {
        {1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f},
        {2.0f/16.0f, 4.0f/16.0f, 2.0f/16.0f},
        {1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f}
    };

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int dy = -1; dy <= 1; ++dy) {
                int py = clamp(y + dy, height);
                for (int dx = -1; dx <= 1; ++dx) {
                    int px = clamp(x + dx, width);
                    sum += in[py * width + px] * k[dy + 1][dx + 1];
                }
            }
            out[y * width + x] = sum;
        }
    }
}

/* Reference Gaussian 5x5 (49 flops/pixel, clamp borders) */
void gauss5_v0_serial(const float* in, float* out, int width, int height, const TuningConfig* cfg) {
    (void)cfg;
    const float inv = 1.0f / 256.0f;
    const float k[5][5] = {
        { 1.0f*inv,  4.0f*inv,  6.0f*inv,  4.0f*inv, 1.0f*inv },
        { 4.0f*inv, 16.0f*inv, 24.0f*inv, 16.0f*inv, 4.0f*inv },
        { 6.0f*inv, 24.0f*inv, 36.0f*inv, 24.0f*inv, 6.0f*inv },
        { 4.0f*inv, 16.0f*inv, 24.0f*inv, 16.0f*inv, 4.0f*inv },
        { 1.0f*inv,  4.0f*inv,  6.0f*inv,  4.0f*inv, 1.0f*inv }
    };

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int dy = -2; dy <= 2; ++dy) {
                int py = clamp(y + dy, height);
                for (int dx = -2; dx <= 2; ++dx) {
                    int px = clamp(x + dx, width);
                    sum += in[py * width + px] * k[dy + 2][dx + 2];
                }
            }
            out[y * width + x] = sum;
        }
    }
}

/* Reference Sobel filter: Gx, Gy, Magnitude (16 flops/pixel, clamp borders) */
void sobel_v0_serial(const float* in, float* out, int width, int height, const TuningConfig* cfg) {
    (void)cfg;
    for (int y = 0; y < height; ++y) {
        int ym1 = clamp(y - 1, height);
        int yp1 = clamp(y + 1, height);
        for (int x = 0; x < width; ++x) {
            int xm1 = clamp(x - 1, width);
            int xp1 = clamp(x + 1, width);

            float p00 = in[ym1 * width + xm1];
            float p01 = in[ym1 * width + x  ];
            float p02 = in[ym1 * width + xp1];

            float p10 = in[y   * width + xm1];
            float p12 = in[y   * width + xp1];

            float p20 = in[yp1 * width + xm1];
            float p21 = in[yp1 * width + x  ];
            float p22 = in[yp1 * width + xp1];

            float gx = (p02 + 2.0f * p12 + p22) - (p00 + 2.0f * p10 + p20);
            float gy = (p20 + 2.0f * p21 + p22) - (p00 + 2.0f * p01 + p02);

            out[y * width + x] = sqrtf(gx * gx + gy * gy);
        }
    }
}

/* Reference 3x3 Laplacian (6 flops/pixel, clamp borders) */
void laplace_v0_serial(const float* in, float* out, int width, int height, const TuningConfig* cfg) {
    (void)cfg;
    for (int y = 0; y < height; ++y) {
        int ym1 = clamp(y - 1, height);
        int yp1 = clamp(y + 1, height);
        for (int x = 0; x < width; ++x) {
            int xm1 = clamp(x - 1, width);
            int xp1 = clamp(x + 1, width);

            float top    = in[ym1 * width + x];
            float bottom = in[yp1 * width + x];
            float left   = in[y * width + xm1];
            float right  = in[y * width + xp1];
            float center = in[y * width + x];

            out[y * width + x] = top + bottom + left + right - 4.0f * center;
        }
    }
}

