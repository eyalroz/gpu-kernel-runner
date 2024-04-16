/*
 * Copyright 2017-2020 NVIDIA Corporation.  All rights reserved.
 * Copyright 2024 GE Healthcare.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
#pragma once
#ifndef CUDA_MMA_SMALLER_HPP_
#define CUDA_MMA_SMALLER_HPP_

#include <mma.h>

#define __CUDA_MMA_DEVICE_DECL__ static __device__ __inline__

#if defined(__cplusplus) && defined(__CUDACC__)

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700


#ifndef __CUDA_ARCH__
#define __DEF_IF_HOST { }
#else  /* !__CUDA_ARCH__ */
#define __DEF_IF_HOST ;
#endif /* __CUDA_ARCH__ */

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 720
#define __CUDA_IMMA__ 1
#endif  /* !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 720 */

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 730
#define __CUDA_SUBBYTE_IMMA__ 1
#endif  /* !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 730 */

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define __CUDA_AMPERE_MMA__ 1
#endif  /* !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800 */

namespace nvcuda {

// Extra definitions in addition to the ones provided by NVIDIA
namespace wmma {
  
  // 
  // Fragment template is:
  // 
  // template<typename Use, int m, int n, int k, typename T, typename Layout=void> class fragment;

  // 
  // Fragments for 16x8x16
  // 
  template<> class fragment<matrix_a, 16, 8, 16, __half, row_major> : public __frag_base<__half, 8> {};
  template<> class fragment<matrix_a, 16, 8, 16, __half, col_major> : public __frag_base<__half, 8> {};
  template<> class fragment<matrix_b, 16, 8, 16, __half, row_major> : public __frag_base<__half, 4> {};
  template<> class fragment<matrix_b, 16, 8, 16, __half, col_major> : public __frag_base<__half, 4> {};
  template<> class fragment<accumulator, 16, 8, 16, __half> : public __frag_base<__half, 4> {};

  // 
  // Fragments for 8x8x4
  // (and remember these are fragments of 4 different matrix triplets, as 4 distinct MMA operations are
  // performed)
  // 
  template<> class fragment<matrix_a, 8, 8, 4, __half, row_major> : public __frag_base<__half, 4> {};
  template<> class fragment<matrix_a, 8, 8, 4, __half, col_major> : public __frag_base<__half, 4> {};
  template<> class fragment<matrix_b, 8, 8, 4, __half, row_major> : public __frag_base<__half, 4> {};
  template<> class fragment<matrix_b, 8, 8, 4, __half, col_major> : public __frag_base<__half, 4> {};
  template<> class fragment<accumulator, 8, 8, 4, __half> : public __frag_base<__half, 8> {};

  // 
  // Fragments for 16x8x8
  // 
  template<> class fragment<matrix_a, 16, 8, 8, __half, row_major> : public __frag_base<__half, 4> {};
  template<> class fragment<matrix_a, 16, 8, 8, __half, col_major> : public __frag_base<__half, 4> {};
  template<> class fragment<matrix_b, 16, 8, 8, __half, row_major> : public __frag_base<__half, 2> {};
  template<> class fragment<matrix_b, 16, 8, 8, __half, col_major> : public __frag_base<__half, 2> {};
  template<> class fragment<accumulator, 16, 8, 8, __half> : public __frag_base<__half, 4> {};

  // 
  // Load functions for frags of shape m16n8k16
  // 
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 8, 16, __half, row_major>& a, const __half* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 8, 16, __half, col_major>& a, const __half* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 8, 16, __half, row_major>& a, const __half* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 8, 16, __half, col_major>& a, const __half* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 16, 8, 16, __half>& a, const __half* p, unsigned ldm, layout_t layout) __DEF_IF_HOST

  //
  // Load functions for frags of shape m8n8k4
  // 
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 8, 4, __half, row_major>& a, const __half* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 8, 4, __half, col_major>& a, const __half* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 8, 4, __half, row_major>& a, const __half* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 8, 4, __half, col_major>& a, const __half* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 8, 8, 4, __half>& a, const __half* p, unsigned ldm, layout_t layout) __DEF_IF_HOST

  //
  // Load functions for frags of shape m16n8k8
  // 
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 8, 8, __half, row_major>& a, const __half* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 8, 8, __half, col_major>& a, const __half* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 8, 8, __half, row_major>& a, const __half* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 8, 8, __half, col_major>& a, const __half* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 16, 8, 8, __half>& a, const __half* p, unsigned ldm, layout_t layout) __DEF_IF_HOST
  
  // 
  // Store functions for frags of shape m16n8k16
  // 
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(__half *p, const fragment<accumulator, 16, 8, 16, __half>& a, unsigned ldm, layout_t layout) __DEF_IF_HOST
  // 
  // Store functions for frags of shape m8n8k4
  // 
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(__half *p, const fragment<accumulator, 8, 8, 4, __half>& a, unsigned ldm, layout_t layout) __DEF_IF_HOST
  // 
  // Store functions for frags of shape m16n8k8
  // 
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(__half *p, const fragment<accumulator, 16, 8, 8, __half>& a, unsigned ldm, layout_t layout) __DEF_IF_HOST

  // 
  // MMA functions for shape m16n8k16
  // 
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 8, 16, __half>& d, const fragment<matrix_a, 16, 8, 16, __half, row_major>& a, const fragment<matrix_b,16, 8, 16, __half, col_major>& b, const fragment<accumulator,16, 8, 16, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 8, 16, __half>& d, const fragment<matrix_a, 16, 8, 16, __half, col_major>& a, const fragment<matrix_b,16, 8, 16, __half, col_major>& b, const fragment<accumulator,16, 8, 16, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 8, 16, __half>& d, const fragment<matrix_a, 16, 8, 16, __half, row_major>& a, const fragment<matrix_b,16, 8, 16, __half, row_major>& b, const fragment<accumulator,16, 8, 16, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 8, 16, __half>& d, const fragment<matrix_a, 16, 8, 16, __half, col_major>& a, const fragment<matrix_b,16, 8, 16, __half, row_major>& b, const fragment<accumulator,16, 8, 16, __half>& c) __DEF_IF_HOST

  // 
  // MMA functions for shape m8n8k4
  // 
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 8, 4, __half>& d, const fragment<matrix_a, 8, 8, 4, __half, row_major>& a, const fragment<matrix_b, 8, 8, 4, __half, col_major>& b, const fragment<accumulator, 8, 8, 4, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 8, 4, __half>& d, const fragment<matrix_a, 8, 8, 4, __half, col_major>& a, const fragment<matrix_b, 8, 8, 4, __half, col_major>& b, const fragment<accumulator, 8, 8, 4, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 8, 4, __half>& d, const fragment<matrix_a, 8, 8, 4, __half, row_major>& a, const fragment<matrix_b, 8, 8, 4, __half, row_major>& b, const fragment<accumulator, 8, 8, 4, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 8, 4, __half>& d, const fragment<matrix_a, 8, 8, 4, __half, col_major>& a, const fragment<matrix_b, 8, 8, 4, __half, row_major>& b, const fragment<accumulator, 8, 8, 4, __half>& c) __DEF_IF_HOST

  // 
  // MMA functions for shape m16n8k8
  // 
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 16, 8, 8, __half>& d, const fragment<matrix_a, 16, 8, 8, __half, row_major>& a, const fragment<matrix_b, 16, 8, 8, __half, col_major>& b, const fragment<accumulator, 16, 8, 8, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 16, 8, 8, __half>& d, const fragment<matrix_a, 16, 8, 8, __half, col_major>& a, const fragment<matrix_b, 16, 8, 8, __half, col_major>& b, const fragment<accumulator, 16, 8, 8, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 16, 8, 8, __half>& d, const fragment<matrix_a, 16, 8, 8, __half, row_major>& a, const fragment<matrix_b, 16, 8, 8, __half, row_major>& b, const fragment<accumulator, 16, 8, 8, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 16, 8, 8, __half>& d, const fragment<matrix_a, 16, 8, 8, __half, col_major>& a, const fragment<matrix_b, 16, 8, 8, __half, row_major>& b, const fragment<accumulator, 16, 8, 8, __half>& c) __DEF_IF_HOST
  
};
};

#undef __DEF_IF_HOST
#undef __CUDA_IMMA__
#undef __CUDA_SUBBYTE_IMMA__
#undef __CUDA_AMPERE_MMA__
#endif /* !__CUDA_ARCH__ || __CUDA_ARCH__ >= 700 */

#endif /* __cplusplus && __CUDACC__ */

#undef __CUDA_MMA_DEVICE_DECL__

#if defined(__CUDA_ARCH__)
#include "mma-smaller.hpp"
#endif /* defined(__CUDA_ARCH__) */

#endif // CUDA_MMA_SMALLER_HPP_
