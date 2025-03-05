#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <riscv_vector.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>

typedef float float32_t;
typedef double float64_t;

#define tensor_new_2d_with_stride(name, _h, _w, _elemsize, _data, stride) \
    Tensor name = { {_h, _w, 0, 0}, 2, _elemsize, _h*_w, _data, stride==0?_w*_elemsize: stride};
    
#define tensor_new_2d(name, _h, _w, _elemsize, _data) \
    tensor_new_2d_with_stride(name, _h, _w, _elemsize, _data, _w*_elemsize) \

typedef struct {
    int shape[4];
    int dims;
    int elemsize;
    int size;
    void *data;
    int stride;
} Tensor;

#endif // __TENSOR_H__