#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <stdint.h>
#include <stddef.h>

#define NOISE_NONE 0
#define NOISE_GAUSSIAN 1
#define NOISE_SALT_PEPPER 2

float* load_image_float(const char* filepath, int* width, int* height);
int save_image_float(const char* filepath, const float* img, int width, int height);
float* allocate_image_float(int width, int height);
void free_image_float(float* img);

void generate_synthetic_image(float* img, int width, int height, int noise_type, float noise_param, unsigned int seed);

#endif /* IMAGE_IO_H */

