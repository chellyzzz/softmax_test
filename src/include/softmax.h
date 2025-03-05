#ifndef _SOFTMAX_H_
#define _SOFTMAX_H_

#include "tensor.h"
#include <riscv_vector.h>
#include <stddef.h>

vfloat32m8_t vfexp_f32m8(vfloat32m8_t vs1, int vl) {
  // assert(vl <= (VLENB * 8 / 4));

  vfloat32m8_t _n = vfmul_vf_f32m8(vs1, ln2_recip_h, vl); // _n = x/ln2_h
  vint32m8_t _n_int32 = vfcvt_x_f_v_i32m8(_n, vl);     //_n_int32 = int(x/ln2_h)
  vfloat32m8_t _n_h = vfcvt_f_x_v_f32m8(_n_int32, vl); //_n_h: float16_t
  vfloat32m8_t _dx =
      vfnmsac_vf_f32m8(vs1, ln2_h, _n_h, vl); // _dx = x - _n_h*ln2

  _n_h = vfadd_vf_f32m8(_n_h, offset_fp32, vl);
  vuint32m8_t _n_uint32_25 = vfcvt_xu_f_v_u32m8(_n_h, vl);
  _n_uint32_25 = vmul_vx_u32m8(_n_uint32_25, 4, vl);

  vfloat32m8_t _der_6 = vloxei32_v_f32m8(der_6, _n_uint32_25, vl);
  vfloat32m8_t _der_2 = vloxei32_v_f32m8(der_2, _n_uint32_25, vl);
  vfloat32m8_t _der_1 = vloxei32_v_f32m8(der_1, _n_uint32_25, vl);

  vfloat32m8_t _res = vfmacc_vv_f32m8(_der_2, _der_6, _dx, vl);
  _res = vfmacc_vv_f32m8(_der_1, _res, _dx, vl);
  _res = vfmacc_vv_f32m8(_der_1, _res, _dx, vl);

  return _res;
}


uint32_t neg_inf_32 = 0xFF800000;

int softmax(Tensor *dst, Tensor *src) {
  assert(dst->size == src->size);

  float32_t *psrc = (float32_t *)src->data;
  float32_t *pdst = (float32_t *)dst->data;

  int vl;

  float32_t neg_inf = *(float32_t *)&neg_inf_32;
  vfloat32m1_t _maxium = vfmv_v_f_f32m1(neg_inf, 1);
  for (int i = 0; i < src->size; i += vl) {
    vl = vsetvl_e32m8(src->size - i);
    vfloat32m8_t _src = vle32_v_f32m8(psrc + i, vl);
    _maxium = vfredmax_vs_f32m8_f32m1(_maxium, _src, _maxium, vl);
  }
  float32_t pmax = vfmv_f_s_f32m1_f32(_maxium);

  vfloat32m1_t _sum = vfmv_v_f_f32m1(0.f, 1);

  for (int i = 0; i < src->size; i += vl) {
    vl = vsetvl_e32m8(src->size - i);
    vfloat32m8_t _src = vle32_v_f32m8(psrc + i, vl);
    vfloat32m8_t _diff = vfsub_vf_f32m8(_src, pmax, vl); // src-max
    vfloat32m8_t _exp = vfexp_f32m8(_diff, vl);
    vse32_v_f32m8(pdst + i, _exp, vl);
    _sum = vfredosum_vs_f32m8_f32m1(_sum, _exp, _sum, vl);
  }
  float32_t psum = vfmv_f_s_f32m1_f32(_sum);
  asm("fence");

  for (int i = 0; i < src->size; i += vl) {
    vl = vsetvl_e32m8(src->size - i);
    vfloat32m8_t _src = vle32_v_f32m8(pdst + i, vl);
    vfloat32m8_t _prob = vfdiv_vf_f32m8(_src, psum, vl);
    vse32_v_f32m8(pdst + i, _prob, vl);
  }

  return 0;
}

#endif // _SOFTMAX_H_