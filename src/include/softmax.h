#ifndef _SOFTMAX_H_
#define _SOFTMAX_H_

#include "tensor.h"
#include <riscv_vector.h>
#include <stddef.h>
#include "exp.h"


uint16_t neg_inf_16 = 0xFC00;

int softmax_e16(Tensor *dst, Tensor *src) {
  // assert(dst->size == src->size);

  float16_t *psrc = (float16_t *)src->data;
  float16_t *pdst = (float16_t *)dst->data;

  int vl;

  float16_t neg_inf = *(float16_t *)&neg_inf_16;
  vfloat16m1_t _maxium = __riscv_vfmv_v_f_f16m1(neg_inf, 1);

  for (int i = 0; i < src->size; i += vl) {
    vl =  __riscv_vsetvl_e16m1(src->size - i);
    vfloat16m1_t _src = __riscv_vle16_v_f16m1(psrc + i, vl);
    _maxium = __riscv_vfredmax_vs_f16m1_f16m1(_src, _maxium, vl);
  }
  float16_t pmax = __riscv_vfmv_f_s_f16m1_f16(_maxium);

  vfloat16m1_t _sum = __riscv_vfmv_v_f_f16m1(0.f, 1);

  for (int i = 0; i < src->size; i += vl) {
    vl =  __riscv_vsetvl_e16m1(src->size - i);
    vfloat16m1_t _src = __riscv_vle16_v_f16m1(psrc + i, vl);
    vfloat16m1_t _diff = __riscv_vfsub_vf_f16m1(_src, pmax, vl); // src-max
    vfloat16m1_t _exp = vfexp_f16m1(_diff, vl);
    __riscv_vse16_v_f16m1(pdst + i, _exp, vl);
    _sum = __riscv_vfredosum_vs_f16m1_f16m1(_exp, _sum, vl);
  }
  float16_t psum = __riscv_vfmv_f_s_f16m1_f16(_sum);
  asm("fence");

  for (int i = 0; i < src->size; i += vl) {
    vl =  __riscv_vsetvl_e16m1(src->size - i);
    vfloat16m1_t _src = __riscv_vle16_v_f16m1(pdst + i, vl);
    vfloat16m1_t _prob = __riscv_vfdiv_vf_f16m1(_src, psum, vl);
    __riscv_vse16_v_f16m1(pdst + i, _prob, vl);
  }

  return 0;
}

#endif // _SOFTMAX_H_
