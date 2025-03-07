#include "tensor.h"
#include <stddef.h>
#include <riscv_vector.h>
#include "softmax.h"
#include "input_data.h"
#include "softmax_rvv.h"

#define H 64
#define W 64
#define NLOOPS 10


int main(int argc, char **argv)
{
    float src[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    size_t n = sizeof(src) / sizeof(src[0]);
    float dst[n];

    softmax_stable_rvv_fp32(dst, src, n);
    
    return 0;
}