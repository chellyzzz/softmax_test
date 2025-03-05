#ifndef _SOFTMAX_H_
#define _SOFTMAX_H_

#include "tensor.h"
#include <riscv_vector.h>
#include <stddef.h>
#include "exp.h"

uint32_t neg_inf_32 = 0xFF800000;

int softmax(Tensor *dst, Tensor *src) {
  // assert(dst->size == src->size);

  float32_t *psrc = (float32_t *)src->data;
  float32_t *pdst = (float32_t *)dst->data;

  int vl;

  float32_t neg_inf = *(float32_t *)&neg_inf_32;
  vfloat32m1_t _maxium = __riscv_vfmv_v_f_f32m1(neg_inf, 1);
  for (int i = 0; i < src->size; i += vl) {
    vl =  __riscv_vsetvl_e32m8(src->size - i);
    vfloat32m8_t _src = __riscv_vle32_v_f32m8(psrc + i, vl);
    _maxium = __riscv_vfredmax_vs_f32m8_f32m1(_src, _maxium, vl);
  }
  float32_t pmax = __riscv_vfmv_f_s_f32m1_f32(_maxium);

  vfloat32m1_t _sum = __riscv_vfmv_v_f_f32m1(0.f, 1);

  for (int i = 0; i < src->size; i += vl) {
    vl =  __riscv_vsetvl_e32m8(src->size - i);
    vfloat32m8_t _src = __riscv_vle32_v_f32m8(psrc + i, vl);
    vfloat32m8_t _diff = __riscv_vfsub_vf_f32m8(_src, pmax, vl); // src-max
    vfloat32m8_t _exp = vfexp_f32m8(_diff, vl);
    __riscv_vse32_v_f32m8(pdst + i, _exp, vl);
    _sum = __riscv_vfredosum_vs_f32m8_f32m1(_exp, _sum, vl);
  }
  float32_t psum = __riscv_vfmv_f_s_f32m1_f32(_sum);
  asm("fence");

  for (int i = 0; i < src->size; i += vl) {
    vl =  __riscv_vsetvl_e32m8(src->size - i);
    vfloat32m8_t _src = __riscv_vle32_v_f32m8(pdst + i, vl);
    vfloat32m8_t _prob = __riscv_vfdiv_vf_f32m8(_src, psum, vl);
    __riscv_vse32_v_f32m8(pdst + i, _prob, vl);
  }

  return 0;
}

#endif // _SOFTMAX_H_
