#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_FAILURE_STRINGS
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "image_io.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float* allocate_image_float(int width, int height) {
    size_t size = (size_t)width * (size_t)height * sizeof(float);
    return (float*)hpc_aligned_alloc(ALIGNMENT_BYTES, size);
}

void free_image_float(float* img) {
    if (img) hpc_aligned_free(img);
}

float* load_image_float(const char* filepath, int* width, int* height) {
    int w, h, channels;
    /* Force 1-channel grayscale */
    stbi_uc* data = stbi_load(filepath, &w, &h, &channels, 1);
    if (!data) {
        fprintf(stderr, "Failed to load image: %s\n", filepath);
        return NULL;
    }
    *width = w;
    *height = h;

    float* img = allocate_image_float(w, h);
    if (!img) {
        stbi_image_free(data);
        return NULL;
    }

    for (int i = 0; i < w * h; ++i) {
        img[i] = (float)data[i];
    }
    stbi_image_free(data);
    return img;
}

int save_image_float(const char* filepath, const float* img, int width, int height) {
    uint8_t* u8_data = (uint8_t*)malloc((size_t)width * (size_t)height);
    if (!u8_data) return 0;

    for (size_t i = 0; i < (size_t)width * (size_t)height; ++i) {
        float val = img[i];
        if (val < 0.0f) val = 0.0f;
        if (val > 255.0f) val = 255.0f;
        u8_data[i] = (uint8_t)(val + 0.5f);
    }

    int ret = stbi_write_png(filepath, width, height, 1, u8_data, width);
    free(u8_data);
    return ret;
}

/* Linear Congruential Generator for reproducible pseudo-random numbers */
static inline uint32_t lcg_next(uint32_t* state) {
    *state = (*state * 1664525u + 1013904223u);
    return *state;
}

static inline float lcg_uniform(uint32_t* state) {
    return (float)(lcg_next(state)) / 4294967296.0f;
}

void generate_synthetic_image(float* img, int width, int height, int noise_type, float noise_param, unsigned int seed) {
    uint32_t rng = seed ? (uint32_t)seed : 123456789u;

    /* Base test image: geometric shapes, gradient, and concentric rings */
    float cx = width * 0.5f;
    float cy = height * 0.5f;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float dx = (float)x - cx;
            float dy = (float)y - cy;
            float r = sqrtf(dx * dx + dy * dy);

            /* Concentric rings + diagonal ramp */
            float base = 128.0f + 64.0f * sinf(r * 0.05f) + 32.0f * ((float)(x + y) / (float)(width + height));
            
            /* Checkerboard grid of 64x64 blocks */
            if (((x / 64) ^ (y / 64)) & 1) {
                base += 20.0f;
            }

            img[y * width + x] = base;
        }
    }

    if (noise_type == NOISE_GAUSSIAN) {
        /* noise_param is sigma (std dev), e.g. 25.0f */
        float sigma = noise_param > 0.0f ? noise_param : 25.0f;
        for (size_t i = 0; i < (size_t)width * (size_t)height; i += 2) {
            /* Box-Muller transform */
            float u1 = lcg_uniform(&rng);
            float u2 = lcg_uniform(&rng);
            if (u1 <= 1e-7f) u1 = 1e-7f;

            float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
            float z1 = sqrtf(-2.0f * logf(u1)) * sinf(2.0f * 3.14159265f * u2);

            img[i] += z0 * sigma;
            if (i + 1 < (size_t)width * (size_t)height) {
                img[i + 1] += z1 * sigma;
            }
        }
    } else if (noise_type == NOISE_SALT_PEPPER) {
        /* noise_param is probability (e.g. 0.05f) */
        float p = noise_param > 0.0f ? noise_param : 0.05f;
        for (size_t i = 0; i < (size_t)width * (size_t)height; ++i) {
            float r = lcg_uniform(&rng);
            if (r < p * 0.5f) {
                img[i] = 0.0f;     /* Pepper */
            } else if (r < p) {
                img[i] = 255.0f;   /* Salt */
            }
        }
    }

    /* Clamp to [0, 255] */
    for (size_t i = 0; i < (size_t)width * (size_t)height; ++i) {
        if (img[i] < 0.0f) img[i] = 0.0f;
        if (img[i] > 255.0f) img[i] = 255.0f;
    }
}
