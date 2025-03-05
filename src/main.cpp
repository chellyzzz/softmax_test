#include "tensor.h"
#include <stddef.h>
#include <riscv_vector.h>
#include "softmax.h"

#define H 64
#define W 64
#define DEBUG_PRINT 1
#define NLOOPS 10

float srcData[H * W];
float dstData[H * W];

int main(int argc, char **argv)
{

    printf("Begin\n");

    int h = H;
    int w = W;

    if (DEBUG_PRINT) {
        printf("In Shape:\n\t(h, w) = (%d, %d)\n",
                    h, w);
    }

    srand((unsigned int)time(NULL));
    for (int i = 0; i < H * W; i++) {
        srcData[i] = (float)(rand() % 1000) / 100.0f; // 0.00 到 10.00 之间
    }

    tensor_new_2d(srcMat, H, W, sizeof(float32_t), srcData);
    tensor_new_2d(dstMat, H, W, sizeof(float32_t), &dstData);


    for (int i = 0; i < NLOOPS; i++) {
        softmax(&dstMat, &srcMat);
    }

    if (DEBUG_PRINT) {
        printf("Softmax Output:\n");
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                printf("%.6f ", dstData[i * W + j]);
            }
            printf("\n");
        }
    }
    
    printf("End\n");

    return 0;
}