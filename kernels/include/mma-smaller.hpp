/*
 * Copyright 2017-2020 NVIDIA Corporation.  All rights reserved.
 * Copyright 2024 Eyal Rozenberg <eyalroz1@gmx.com>.
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

#ifndef CUDA_MMA_SMALLER_CUH_
#error "Do not include this file independently of 'cuda-mma-smaller.cuh'."
#endif

// Include this from within mma-smaller.cuh !

#if defined(__cplusplus) && defined(__CUDACC__)

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "mma-smaller-intrinsics.hpp"


#define __CUDA_MMA_DEVICE_DECL__ static __device__ __inline__

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
namespace wmma {

namespace detail {

  template <typename T>
  T remove_ref_helper(T&);

  template <typename T>
  T remove_ref_helper(T&&);


__CUDA_MMA_DEVICE_DECL__ unsigned lane_id() {
  // We can use the special register...
  //
  // unsigned ret;
  // asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
  // return ret;s
  //
  // ... or just be more straightforward (and assume the X dimension has full warps)
  enum { warp_size = 32 };
  constexpr const auto lane_id_mask = warp_size - 1;
  return threadIdx.x & lane_id_mask;  
}

/* // an inclusive integer range
struct range { int first, last; }

bool is_in(int i, range r) { return i >= r.first and i <= r.last; }
bool is_in_one_of(int i, range r1, range r2) { return is_in(i, r1) or is_in(i, r2); }
*/

} // namespace detail

  // 
  // Load functions for fragments of shape m16n8k16
  // 
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 8, 16, __half, row_major>& a, const __half* p, unsigned ldm) 
  {
    // Note: by default and with no padding/alignment, we should have
    // set ldm = is_row_major ? num_cols : num_rows; but NVIDIA mandates
    // we take this value explicitly.
    using fragment_type = decltype(detail::remove_ref_helper(a));
    static constexpr const bool is_row_major = true;
    enum { M = 16, N = 8, K = 16 };
    enum { num_rows = M, num_cols = K };
    auto lane_id = detail::lane_id();

    auto groupID = lane_id >> 2;
    auto threadID_in_group = lane_id % 4;

    auto base_row = groupID;
    auto base_col = (threadID_in_group * 2);

    #pragma unroll
    for(int i = 0; i < fragment_type::num_elements; i++) {
      auto row = base_row + (((i & ~0x4) < 2) ? 0 : 8);
      auto col = base_col + (i & 0x1) + ((i < 4) ? 0 : 8);
      auto pos = is_row_major ?
        (row * ldm + col) :
        (col * ldm + row);
//      if (threadIdx.x == 8) printf("(%3d,%3d,%3d): a.x[%3d] = %.1f, (row,col) = (%3d,%3d) pos is %3d;\n",
//             threadIdx.x, threadIdx.y, threadIdx.z, i, (float) p[pos], row, col, (int) pos);

      //      printf("(%3d,%3d,%3d): Lane %2d group %1d in_group %d base_row %d; for a.x[%3d], (row,col) = (%2d,%2d) and my pos is %3d; i & 0x1 = %d\n",
//             threadIdx.x, threadIdx.y, threadIdx.z,
//             lane_id, groupID, threadID_in_group, base_row, i, row, col, (int) pos, (int)(i & 0x1));
      a.x[i] = p[pos];
    }
  } 

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 8, 16, __half, col_major>& a, const __half* p, unsigned ldm)
  {
    // Note: by default and with no padding/alignment, we should have
    // set ldm = is_row_major ? num_cols : num_rows; but NVIDIA mandates
    // we take this value explicitly.
    using fragment_type = decltype(detail::remove_ref_helper(a));
    static constexpr const bool is_row_major = false;
    enum { M = 16, N = 8, K = 16 };
    enum { num_rows = M, num_cols = K };
    auto lane_id = detail::lane_id();

    auto groupID = lane_id >> 2;
    auto threadID_in_group = lane_id % 4;

    auto base_row = groupID;
    auto base_col = (threadID_in_group * 2);

    #pragma unroll
    for(int i = 0; i < fragment_type::num_elements; i++) {
      auto row = base_row + (((i & ~0x4) < 2) ? 0 : 8);
      auto col = base_col + (i & 0x1) + ((i < 4) ? 0 : 8);
      auto pos = is_row_major ?
        (row * ldm + col) :
        (col * ldm + row);
      a.x[i] = p[pos];
    }
  } 

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b,16, 8, 16, __half, row_major>& a, const __half* p, unsigned ldm)
  {
    // Note: by default and with no padding/alignment, we should have
    // set ldm = is_row_major ? num_cols : num_rows; but NVIDIA mandates
    // we take this value explicitly.
    using fragment_type = decltype(detail::remove_ref_helper(a));
    static constexpr const bool is_row_major = true;
    enum { M = 16, N = 8, K = 16 };
    enum { num_rows = K, num_cols = N };
    auto lane_id = detail::lane_id();

    auto groupID = lane_id >> 2;
    auto threadID_in_group = lane_id % 4;

    auto base_row = (threadID_in_group * 2);
    auto base_col = groupID;

    #pragma unroll
    for(int i = 0; i < fragment_type::num_elements; i++) {
      auto row = base_row + (i & 0x1) + ((i < 2) ? 0 : 8);
      auto col = base_col;
      auto pos = is_row_major ?
        row * ldm + col :
        col * ldm + row;
      a.x[i] = p[pos];
//      printf("(%2d,%2d,%2d): a.x[%2d] = %.1f, (row,col) = (%3d,%3d) pos is %3d;\n",
//             threadIdx.x, threadIdx.y, threadIdx.z, i, (float) p[pos], row, col, (int) pos);
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b,16, 8, 16, __half, col_major>& a, const __half* p, unsigned ldm) 
  {
    // Note: by default and with no padding/alignment, we should have
    // set ldm = is_row_major ? num_cols : num_rows; but NVIDIA mandates
    // we take this value explicitly.
    using fragment_type = decltype(detail::remove_ref_helper(a));
    static constexpr const bool is_row_major = false;
    enum { M = 16, N = 8, K = 16 };
    enum { num_rows = K, num_cols = N };
    auto lane_id = detail::lane_id();

    auto groupID = lane_id >> 2;
    auto threadID_in_group = lane_id % 4;

    auto base_row = (threadID_in_group * 2);
    auto base_col = groupID;

    #pragma unroll
    for(int i = 0; i < fragment_type::num_elements; i++) {
      auto row = base_row + (i & 0x1) + ((i < 2) ? 0 : 8);
      auto col = base_col;
      auto pos = is_row_major ?
        row * ldm + col :
        col * ldm + row;
      a.x[i] = p[pos];
//      printf("(%2d,%2d,%2d): a.x[%2d] = %5.0f, (row,col) = (%3d,%3d) pos is %3d;\n",
//             threadIdx.x, threadIdx.y, threadIdx.z, i, (float) p[pos], row, col, (int) pos);
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator,16, 8, 16, __half>& a, const __half* p, unsigned ldm, layout_t layout) 
  {
    // Note: by default and with no padding/alignment, we should have
    // set ldm = is_row_major ? num_cols : num_rows; but NVIDIA mandates
    // we take this value explicitly.
    using fragment_type = decltype(detail::remove_ref_helper(a));
    enum { M = 16, N = 8, K = 16 };
    enum { num_rows = M, num_cols = N };
    auto lane_id = detail::lane_id();

    auto groupID = lane_id >> 2;
    auto threadID_in_group = lane_id % 4;

    auto base_row = groupID;
    auto base_col = (threadID_in_group * 2);

    #pragma unroll
    for(int i = 0; i < fragment_type::num_elements; i++) {
      auto row = base_row + ((i < 2) ? 0 : 8);
      auto col = base_col + (i & 0x1);
      auto pos = (layout == mem_row_major) ?
        row * ldm + col :
        col * ldm + row;
      a.x[i] = p[pos];
    }
  }

  // 
  // Load functions for fragments of shape m8n8k4
  // 
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 8, 4, __half, row_major>& a, const __half* p, unsigned ldm) {
    asm("trap;"); // not yet tested
    // Note: NOT like the column-major function
    // TODO: Double-check code involving ldm
    // we take this value explicitly.
    using fragment_type = decltype(detail::remove_ref_helper(a));
    static constexpr const bool is_row_major = true;
    enum { M = 8, N = 8, K = 4 };
    enum { num_rows = M, num_cols = K };
    auto lane_id = detail::lane_id();

    auto matrix_index = (lane_id % 16) >> 2;
        // This MMA variant acts on 4 independent pairs of matrice, which we assume are consecutive in device memory
    auto base_row = lane_id % 4 + ((lane_id < 16) ? 0 : 4);
    auto base_col = 0;

    #pragma unroll
    for(int i = 0; i < fragment_type::num_elements; i++) {
      auto row = base_row;
      auto col = base_col + i;
      auto pos = matrix_index * num_rows * ldm + row * ldm + col;
//      if (threadIdx.x == 8) printf("(%3d,%3d,%3d): a.x[%3d] = %.1f, (row,col) = (%3d,%3d) pos is %3d;\n",
//             threadIdx.x, threadIdx.y, threadIdx.z, i, (float) p[pos], row, col, (int) pos);

      //      printf("(%3d,%3d,%3d): Lane %2d group %1d in_group %d base_row %d; for a.x[%3d], (row,col) = (%2d,%2d) and my pos is %3d; i & 0x1 = %d\n",
//             threadIdx.x, threadIdx.y, threadIdx.z,
//             lane_id, groupID, threadID_in_group, base_row, i, row, col, (int) pos, (int)(i & 0x1));
      a.x[i] = p[pos];
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 8, 4, __half, col_major>& a, const __half* p, unsigned ldm) {
    asm("trap;"); // not yet tested
    // Same as row_major except for is_row_major
    // TODO: Double-check code involving ldm
    using fragment_type = decltype(detail::remove_ref_helper(a));
    enum { M = 8, N = 8, K = 4 };
    enum { num_rows = M, num_cols = K };
    auto lane_id = detail::lane_id();

    auto matrix_index = (lane_id % 16) >> 2;
        // This MMA variant acts on 4 independent pairs of matrice, which we assume are consecutive in device memory
    auto base_row = ((lane_id < 16) ? 0 : 4);
    auto base_col = lane_id % 4;

    #pragma unroll
    for(int i = 0; i < fragment_type::num_elements; i++) {
      auto row = base_row + i % 4;
      auto col = base_col;
      auto pos = matrix_index * num_rows * ldm + row * ldm + col;
//      if (threadIdx.x == 8) printf("(%3d,%3d,%3d): a.x[%3d] = %.1f, (row,col) = (%3d,%3d) pos is %3d;\n",
//             threadIdx.x, threadIdx.y, threadIdx.z, i, (float) p[pos], row, col, (int) pos);

      //      printf("(%3d,%3d,%3d): Lane %2d group %1d in_group %d base_row %d; for a.x[%3d], (row,col) = (%2d,%2d) and my pos is %3d; i & 0x1 = %d\n",
//             threadIdx.x, threadIdx.y, threadIdx.z,
//             lane_id, groupID, threadID_in_group, base_row, i, row, col, (int) pos, (int)(i & 0x1));
      a.x[i] = p[pos];
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 8, 4, __half, row_major>& a, const __half* p, unsigned ldm) {
    asm("trap;");
    // NOTE: NOT LIKE THE COL-MAJOR CASE
    // TODO: Account for ldm
    using fragment_type = decltype(detail::remove_ref_helper(a));
    enum { M = 8, N = 8, K = 4 };
    enum { num_rows = K, num_cols = N };
    auto lane_id = detail::lane_id();

    auto matrix_id = lane_id % 16 >> 2;
    auto base_row = lane_id % 4;
    auto base_col = ((lane_id < 16) ? 0 : 4);

    static_assert(fragment_type::num_elements == K, "Unexpected fragment_type::num_elements - expected K");
    #pragma unroll
    for(int i = 0; i < fragment_type::num_elements; i++) {
      auto row = base_row;
      auto col = base_col + i;
      auto pos = matrix_id * num_rows * ldm + row * ldm + col;
      a.x[i] = p[pos];
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 8, 4, __half, col_major>& a, const __half* p, unsigned ldm) {
    asm("trap;");
    // NOTE: NOT LIKE THE COL-MAJOR CASE
    // TODO: Account for ldm
    using fragment_type = decltype(detail::remove_ref_helper(a));
    enum { M = 8, N = 8, K = 4 };
    enum { num_rows = K, num_cols = N };
    auto lane_id = detail::lane_id();

    auto matrix_id = lane_id % 16 >> 2;
    auto base_row = 0;
    auto base_col = lane_id % 4 + ((lane_id < 16) ? 0 : 4);

    static_assert(fragment_type::num_elements == K, "Unexpected fragment_type::num_elements - expected K");
    #pragma unroll
    for(int i = 0; i < fragment_type::num_elements; i++) {
      auto row = base_row + i;
      auto col = base_col;
      auto pos = matrix_id * num_rows * ldm + row * ldm + col;
      a.x[i] = p[pos];
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 8, 8, 4, __half>& a, const __half* p, unsigned ldm, layout_t layout) {
    asm("trap;"); // not yet tested
    using fragment_type = decltype(detail::remove_ref_helper(a));
    enum { M = 8, N = 8, K = 4 };
    enum { num_rows = M, num_cols = N };
    auto lane_id = detail::lane_id();

    auto matrix_id = lane_id % 16 >> 2;
    auto base_row = lane_id % 4 + ((lane_id < 16) ? 0 : 4);
    auto base_col = 0;

    #pragma unroll
    for(int i = 0; i < fragment_type::num_elements; i++) {
      auto row = base_row;
      auto col = base_col + i;
      auto pos =
          matrix_id * num_rows * num_cols +
          (layout == mem_row_major) ?
              (row * ldm + col) :
              (col * ldm + row);
      a.x[i] = p[pos];
    }
  }

  // 
  // Load functions for fragments of shape m16n8k8
  // 
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 8, 8, __half, row_major>& a, const __half* p, unsigned ldm) {
    // Note: by default and with no padding/alignment, we should have
    // set ldm = is_row_major ? num_cols : num_rows; but NVIDIA mandates
    // we take this value explicitly.
    using fragment_type = decltype(detail::remove_ref_helper(a));
    static constexpr const bool is_row_major = true;
    enum { M = 16, N = 8, K = 8 };
    enum { num_rows = M, num_cols = K };
    auto lane_id = detail::lane_id();
//    if (lane_id == 0) printf("loading acc %dx%d\n", num_rows, num_cols);

    auto groupID = lane_id >> 2;
    auto threadID_in_group = lane_id % 4;

    auto base_row = groupID;
    auto base_col = (threadID_in_group * 2);

    #pragma unroll
    for(int i = 0; i < fragment_type::num_elements; i++) {
      auto row = base_row + (i < 2 ? 0 : 8);
      auto col = base_col + (i & 0x1);
      auto pos = is_row_major ?
        (row * ldm + col) :
        (col * ldm + row);
      a.x[i] = p[pos];
//      if ((double) a.x[i] != 0) printf("(%2d,%2d,%2d): After load acc; a.x[%3d] = %5.0f, (row,col) = (%3d,%3d) pos is %3d;\n",
//             threadIdx.x, threadIdx.y, threadIdx.z, i, (float) p[pos], row, col, (int) pos);
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 8, 8, __half, col_major>& a, const __half* p, unsigned ldm) {
    // Note: by default and with no padding/alignment, we should have
    // set ldm = is_row_major ? num_cols : num_rows; but NVIDIA mandates
    // we take this value explicitly.
    using fragment_type = decltype(detail::remove_ref_helper(a));
    static constexpr const bool is_row_major = false;
    enum { M = 16, N = 8, K = 8 };
    enum { num_rows = M, num_cols = K };
    auto lane_id = detail::lane_id();

    auto groupID = lane_id >> 2;
    auto threadID_in_group = lane_id % 4;

    auto base_row = groupID;
    auto base_col = (threadID_in_group * 2);

    #pragma unroll
    for(int i = 0; i < fragment_type::num_elements; i++) {
      auto row = base_row + (i < 2 ? 0 : 8);
      auto col = base_col + (i & 0x1);
      auto pos = is_row_major ?
        (row * ldm + col) :
        (col * ldm + row);
      a.x[i] = p[pos];
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 8, 8, __half, row_major>& a, const __half* p, unsigned ldm) {
    // Note: by default and with no padding/alignment, we should have
    // set ldm = is_row_major ? num_cols : num_rows; but NVIDIA mandates
    // we take this value explicitly.
    using fragment_type = decltype(detail::remove_ref_helper(a));
    static constexpr const bool is_row_major = true;
    enum { M = 16, N = 8, K = 8 };
    enum { num_rows = K, num_cols = N };
    auto lane_id = detail::lane_id();

    auto groupID = lane_id >> 2;
    auto threadID_in_group = lane_id % 4;

    auto base_row = (threadID_in_group * 2);
    auto base_col =  groupID;

    #pragma unroll
    for(int i = 0; i < fragment_type::num_elements; i++) {
      auto row = base_row + i;
      auto col = base_col;
      auto pos = is_row_major ?
        (row * ldm + col) :
        (col * ldm + row);
      a.x[i] = p[pos];
//      printf("(%2d,%2d,%2d) Load B row-major: a.x[%2d] = %5.0f, (row,col) = (%3d,%3d) pos is %3d;\n",
//             threadIdx.x, threadIdx.y, threadIdx.z, i, (float) p[pos], row, col, (int) pos);
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 8, 8, __half, col_major>& a, const __half* p, unsigned ldm) {
    // Note: by default and with no padding/alignment, we should have
    // set ldm = is_row_major ? num_cols : num_rows; but NVIDIA mandates
    // we take this value explicitly.
    using fragment_type = decltype(detail::remove_ref_helper(a));
    static constexpr const bool is_row_major = false;
    enum { M = 16, N = 8, K = 8 };
    enum { num_rows = K, num_cols = N };
    auto lane_id = detail::lane_id();
//    if (lane_id == 0) printf("loading B %dx%d\n", num_rows, num_cols);

    auto groupID = lane_id >> 2;
    auto threadID_in_group = lane_id % 4;

    auto base_row = (threadID_in_group * 2);
    auto base_col =  groupID;

    #pragma unroll
    for(int i = 0; i < fragment_type::num_elements; i++) {
      auto row = base_row + i;
      auto col = base_col;
      auto pos = is_row_major ?
        (row * ldm + col) :
        (col * ldm + row);
      a.x[i] = p[pos];
//      printf("(%2d,%2d,%2d) Load B col-major: a.x[%2d] = %5.0f, (row,col) = (%3d,%3d) pos is %3d;\n",
//             threadIdx.x, threadIdx.y, threadIdx.z, i, (float) p[pos], row, col, (int) pos);
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 16, 8, 8, __half>& a, const __half* p, unsigned ldm, layout_t layout) {
    // Note: by default and with no padding/alignment, we should have
    // set ldm = is_row_major ? num_cols : num_rows; but NVIDIA mandates
    // we take this value explicitly.
    using fragment_type = decltype(detail::remove_ref_helper(a));
    enum { M = 16, N = 8, K = 8 };
    enum { num_rows = M, num_cols = N };
    auto lane_id = detail::lane_id();
//    if (lane_id == 0) printf("loading C %dx%d with layout %s-major\n", num_rows, num_cols, (layout == mem_row_major) ? "row" : "col");

    auto groupID = lane_id >> 2;
    auto threadID_in_group = lane_id % 4;

    auto base_row = groupID;
    auto base_col = (threadID_in_group * 2);

    #pragma unroll
    for(int i = 0; i < fragment_type::num_elements; i++) {
      auto row = base_row + (i < 2 ? 0 : 8);
      auto col = base_col + (i & 0x1);
      auto pos = (layout == mem_row_major) ?
        row * ldm + col :
        col * ldm + row;
      a.x[i] = p[pos];
//      if ((double) a.x[i] != 0) printf("(%2d,%2d,%2d): After load of acc; a.x[%3d] = %5.0f, (row,col) = (%3d,%3d) pos is %3d;\n",
//             threadIdx.x, threadIdx.y, threadIdx.z, i, (float) p[pos], row, col, (int) pos);
//      if (lane_id == 0) printf("(%2d,%2d,%2d): After load of acc; a.x[%3d] = %5.0f, (row,col) = (%3d,%3d) pos is %3d;\n",
//             threadIdx.x, threadIdx.y, threadIdx.z, i, (float) p[pos], row, col, (int) pos);
//      if (lane_id == 0) {
//          for (int i1 = 0; i1 < 16; i1++) {
//            for (int j1 = 0; j1 < 8; j1++) {
//                auto pos1 = i1*num_rows + j1;
//                if ((double) p[pos] != 0.f) printf("%2d %2d p[%3d] = %4.f\n", i1, j1, pos1, (double) p[pos1]);
//            }
//        }
//      }
    }
  }

  // 
  // Store functions for fragments of shape m16n8k16
  // 
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(__half *p, const fragment<accumulator,16, 8, 16, __half>& a, unsigned ldm, layout_t layout) {
    // Note: by default and with no padding/alignment, we should have
    // set ldm = is_row_major ? num_cols : num_rows; but NVIDIA mandates
    // we take this value explicitly.
    using fragment_type = decltype(detail::remove_ref_helper(a));
    enum { M = 16, N = 8, K = 16 };
    enum { num_rows = M, num_cols = N };
    auto lane_id = detail::lane_id();

    auto groupID = lane_id >> 2;
    auto threadID_in_group = lane_id % 4;

    auto base_row = groupID;
    auto base_col = (threadID_in_group * 2);

//    if (lane_id == 0) {
//        printf ("mem_row_major ? %s\n", (layout == mem_row_major) ? "true" : "false" );
//        printf ("fragment<accumulator,16, 8, 16, __half>::num_elements = %d\n", (int) fragment_type::num_elements);
//    }
    #pragma unroll
    for(int i = 0; i < fragment_type::num_elements; i++) {
      auto row = base_row + ((i < 2) ? 0 : 8);
      auto col = base_col + (i & 0x1);
      auto pos = (layout == mem_row_major) ?
        row * ldm + col :
        col * ldm + row;
      p[pos] = a.x[i]; // This is the only line differing from the load operation
//      if (lane_id == 4) printf("(%2d,%2d,%2d): Store of Acc; a.x[%3d] = %5.0f, (row,col) = (%3d,%3d) pos is %3d;\n",
//             threadIdx.x, threadIdx.y, threadIdx.z, i, (float) p[pos], row, col, (int) pos);
    }
  }

  // 
  // Store functions for fragments of shape m8n8k4
  // 
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(__half *p, const fragment<accumulator, 8, 8, 4, __half>& a, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major)
      asm("trap;");
    else
      asm("trap;");
  }

  // 
  // Store functions for fragments of shape m16n8k8
  // 
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(__half *p, const fragment<accumulator, 16, 8, 8, __half>& a, unsigned ldm, layout_t layout) {
    // Note: by default and with no padding/alignment, we should have
    // set ldm = is_row_major ? num_cols : num_rows; but NVIDIA mandates
    // we take this value explicitly.
    using fragment_type = decltype(detail::remove_ref_helper(a));
    enum { M = 16, N = 8, K = 8 };
    enum { num_rows = M, num_cols = N };
    auto lane_id = detail::lane_id();
//    if (lane_id == 0) printf("loading D %dx%d with layout %s-major\n", num_rows, num_cols, (layout == mem_row_major) ? "row" : "col");
    __syncthreads();

    auto groupID = lane_id >> 2;
    auto threadID_in_group = lane_id % 4;

    auto base_row = groupID;
    auto base_col = (threadID_in_group * 2);

//    if (lane_id == 0) {
//        printf ("mem_row_major ? %s\n", (layout == mem_row_major) ? "true" : "false" );
//        printf ("fragment<accumulator,16, 8, 16, __half>::num_elements = %d\n", (int) fragment_type::num_elements);
//    }
    #pragma unroll
    for(int i = 0; i < fragment_type::num_elements; i++) {
      auto row = base_row + (i < 2 ? 0 : 8);
      auto col = base_col + i;
      auto pos = (layout == mem_row_major) ?
        row * ldm + col :
        col * ldm + row;
      p[pos] = a.x[i];
//      if ((lane_id == 0 && i ==0 )|| (float) a.x[i] != 0) printf("(%2d,%2d,%2d): Store of Acc; a.x[%3d] = %5.0f, (row,col) = (%3d,%3d) pos is %3d;\n",
//             threadIdx.x, threadIdx.y, threadIdx.z, i, (float) p[pos], row, col, (int) pos);
    }
  }

  // 
  // MMA functions for shape m16n8k16
  // 
  // D fp16, C fp16
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 8, 16, __half>& d, const fragment<matrix_a, 16, 8, 16, __half, row_major>& a, const fragment<matrix_b,16, 8, 16, __half, col_major>& b, const fragment<accumulator,16, 8, 16, __half>& c) {
    mma_sync_aligned_m16n8k16_row_col_f16_f16_f16_f16(d.x, a.x, b.x, c.x);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 8, 16, __half>& d, const fragment<matrix_a, 16, 8, 16, __half, col_major>& a, const fragment<matrix_b,16, 8, 16, __half, col_major>& b, const fragment<accumulator,16, 8, 16, __half>& c) {
    asm("trap;"); // Not yet implemented
  }
    
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 8, 16, __half>& d, const fragment<matrix_a, 16, 8, 16, __half, row_major>& a, const fragment<matrix_b,16, 8, 16, __half, row_major>& b, const fragment<accumulator,16, 8, 16, __half>& c) {
    asm("trap;"); // Not yet implemented
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 8, 16, __half>& d, const fragment<matrix_a, 16, 8, 16, __half, col_major>& a, const fragment<matrix_b,16, 8, 16, __half, row_major>& b, const fragment<accumulator,16, 8, 16, __half>& c) {
    asm("trap;"); // Not yet implemented
  }

  // 
  // MMA functions for shape m8n8k4
  // 
  // D fp16, C fp16
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,8, 8, 4, __half>& d, const fragment<matrix_a, 8, 8, 4, __half, row_major>& a, const fragment<matrix_b,8, 8, 4, __half, col_major>& b, const fragment<accumulator,8, 8, 4, __half>& c) {
    mma_sync_aligned_m8n8k4_row_col_f16_f16_f16_f16(d.x, a.x, b.x, c.x);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,8, 8, 4, __half>& d, const fragment<matrix_a, 8, 8, 4, __half, col_major>& a, const fragment<matrix_b,8, 8, 4, __half, col_major>& b, const fragment<accumulator,8, 8, 4, __half>& c) {
    asm("trap;"); // Not yet implemented
  }

  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,8, 8, 4, __half>& d, const fragment<matrix_a, 8, 8, 4, __half, row_major>& a, const fragment<matrix_b,8, 8, 4, __half, row_major>& b, const fragment<accumulator,8, 8, 4, __half>& c) {
    asm("trap;"); // Not yet implemented
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,8, 8, 4, __half>& d, const fragment<matrix_a, 8, 8, 4, __half, col_major>& a, const fragment<matrix_b,8, 8, 4, __half, row_major>& b, const fragment<accumulator,8, 8, 4, __half>& c) {
    asm("trap;"); // Not yet implemented
  }

  // 
  // MMA functions for shape m16n8k8
  // 
  // D fp16, C fp16
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 8, 8, __half>& d, const fragment<matrix_a, 16, 8, 8, __half, row_major>& a, const fragment<matrix_b,16, 8, 8, __half, col_major>& b, const fragment<accumulator,16, 8, 8, __half>& c) {
     mma_sync_aligned_m16n8k8_row_col_f16_f16_f16_f16(d.x, a.x, b.x, c.x);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 8, 8, __half>& d, const fragment<matrix_a, 16, 8, 8, __half, col_major>& a, const fragment<matrix_b,16, 8, 8, __half, col_major>& b, const fragment<accumulator,16, 8, 8, __half>& c) {
    asm("trap;"); // Not yet implemented
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 8, 8, __half>& d, const fragment<matrix_a, 16, 8, 8, __half, row_major>& a, const fragment<matrix_b,16, 8, 8, __half, row_major>& b, const fragment<accumulator,16, 8, 8, __half>& c) {
    asm("trap;"); // Not yet implemented
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 8, 8, __half>& d, const fragment<matrix_a, 16, 8, 8, __half, col_major>& a, const fragment<matrix_b,16, 8, 8, __half, row_major>& b, const fragment<accumulator,16, 8, 8, __half>& c) {
    asm("trap;"); // Not yet implemented
  }

};
};

#undef __CUDA_IMMA__
#undef __CUDA_SUBBYTE_IMMA__
#undef __CUDA_MMA_DEVICE_DECL__
#undef __CUDA_AMPERE_MMA__

#endif /* !__CUDA_ARCH__ || __CUDA_ARCH__ >= 700 */

#endif /* __cplusplus && __CUDACC__ */


#endif // CUDA_MMA_SMALLER_HPP_

