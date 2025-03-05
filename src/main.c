#include "tensor.h"
#include <stddef.h>
#include <riscv_vector.h>
#include "softmax.h"
#include "input_data.h"

#define H 64
#define W 64
#define NLOOPS 10

float dstData[H * W];

int main(int argc, char **argv)
{
    int h = H;
    int w = W;

    // 使用从 input_data.h 导入的随机数据
    tensor_new_2d(srcMat, H, W, sizeof(float32_t), input_data);  // 使用 input_data 替代原来的 srcData
    tensor_new_2d(dstMat, H, W, sizeof(float32_t), dstData);    // 目标数据

    for (int i = 0; i < NLOOPS; i++) {
        softmax(&dstMat, &srcMat);
    }

    return 0;
}