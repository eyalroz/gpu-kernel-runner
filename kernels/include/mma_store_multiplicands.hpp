/**
 * @file
 *
 * @copyright (c) 2024 Eyal Rozenberg <eyalroz1@gmx.com>.
 * @copyright (c) 2024 GE Healthcare.
 *
 * @note may require conditioning on Ampere-or-greater architecture.
 *
 * @license BSD 3-Clause license:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once
#ifndef CUDA_MMA_STORE_MULTIPLICANDS_HPP
#define CUDA_MMA_STORE_MULTIPLICANDS_HPP

#if defined(__cplusplus) && defined(__CUDACC__)

#if defined(__CUDA_ARCH__)
#include "mma_smaller.cuh"
#endif /* defined(__CUDA_ARCH__) */

#include <mma.h>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700

#include <cuda_fp16.h>
#include <cuda_bf16.h>

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

} // namespace detail

//
// Store functions for fragments of shape m16n8k16
//
__CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(__half* p, fragment<matrix_a, 16, 8, 16, __half, row_major> const & a, unsigned ldm)
{
  // TODO: Is it legitimate for us not to take the row-vs-column-major argument (which in mma.hpp is taken explicitly?)

  // TODO: Account for ldm
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
      (row * num_cols + col) :
      (col * num_rows + row);
//      if (threadIdx.x == 8) printf("(%3d,%3d,%3d): a.x[%3d] = %.1f, (row,col) = (%3d,%3d) pos is %3d;\n",
//             threadIdx.x, threadIdx.y, threadIdx.z, i, (float) p[pos], row, col, (int) pos);

    //      printf("(%3d,%3d,%3d): Lane %2d group %1d in_group %d base_row %d; for a.x[%3d], (row,col) = (%2d,%2d) and my pos is %3d; i & 0x1 = %d\n",
//             threadIdx.x, threadIdx.y, threadIdx.z,
//             lane_id, groupID, threadID_in_group, base_row, i, row, col, (int) pos, (int)(i & 0x1));
    p[pos] = a.x[i];
  }
}

__CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(__half* p, fragment<matrix_a, 16, 8, 16, __half, col_major> const & a, unsigned ldm)
{
  // TODO: Account for ldm
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
      (row * num_cols + col) :
      (col * num_rows + row);
    p[pos] = a.x[i];
  }
}

__CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(__half* p, fragment<matrix_b,16, 8, 16, __half, row_major> const & a, unsigned ldm)
{
  // TODO: Account for ldm
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
      row * num_cols + col :
      col * num_rows + row;
    p[pos] = a.x[i];
//      printf("(%2d,%2d,%2d): a.x[%2d] = %.1f, (row,col) = (%3d,%3d) pos is %3d;\n",
//             threadIdx.x, threadIdx.y, threadIdx.z, i, (float) p[pos], row, col, (int) pos);
  }
}

__CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(__half* p, fragment<matrix_b,16, 8, 16, __half, col_major> const & a, unsigned ldm)
{
  // TODO: Account for ldm
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
      row * num_cols + col :
      col * num_rows + row;
    p[pos] = a.x[i];
//      printf("(%2d,%2d,%2d): a.x[%2d] = %5.0f, (row,col) = (%3d,%3d) pos is %3d;\n",
//             threadIdx.x, threadIdx.y, threadIdx.z, i, (float) p[pos], row, col, (int) pos);
  }
}

// Can probably safely delete this
/*

//
// Store functions for fragments of shape m8n8k4
//
__CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(__half* p, fragment<matrix_a, 8, 8, 4, __half, row_major> const & a, unsigned ldm) {
  asm("trap;"); // not yet tested
  // Note: NOT like the column-major function
  // TODO: Account for ldm
  using fragment_type = decltype(detail::remove_ref_helper(a));
  static constexpr const bool is_row_major = true;
  (void) is_row_major;
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
    auto pos = matrix_index * num_rows * num_cols + row * num_cols + col;
//      if (threadIdx.x == 8) printf("(%3d,%3d,%3d): a.x[%3d] = %.1f, (row,col) = (%3d,%3d) pos is %3d;\n",
//             threadIdx.x, threadIdx.y, threadIdx.z, i, (float) p[pos], row, col, (int) pos);

    //      printf("(%3d,%3d,%3d): Lane %2d group %1d in_group %d base_row %d; for a.x[%3d], (row,col) = (%2d,%2d) and my pos is %3d; i & 0x1 = %d\n",
//             threadIdx.x, threadIdx.y, threadIdx.z,
//             lane_id, groupID, threadID_in_group, base_row, i, row, col, (int) pos, (int)(i & 0x1));
    p[pos] = a.x[i];
  }
}

__CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(__half* p, fragment<matrix_a, 8, 8, 4, __half, col_major> const & a, unsigned ldm) {
  asm("trap;"); // not yet tested
  // Same as row_major except for is_row_major
  // TODO: Account for ldm
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
    auto pos = matrix_index * num_rows * num_cols + row * num_cols + col;
//      if (threadIdx.x == 8) printf("(%3d,%3d,%3d): a.x[%3d] = %.1f, (row,col) = (%3d,%3d) pos is %3d;\n",
//             threadIdx.x, threadIdx.y, threadIdx.z, i, (float) p[pos], row, col, (int) pos);

    //      printf("(%3d,%3d,%3d): Lane %2d group %1d in_group %d base_row %d; for a.x[%3d], (row,col) = (%2d,%2d) and my pos is %3d; i & 0x1 = %d\n",
//             threadIdx.x, threadIdx.y, threadIdx.z,
//             lane_id, groupID, threadID_in_group, base_row, i, row, col, (int) pos, (int)(i & 0x1));
    p[pos] = a.x[i];
  }
}

__CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(__half* p, fragment<matrix_b, 8, 8, 4, __half, row_major> const & a, unsigned ldm) {
  asm("trap;");
  // NOTE: NOT LIKE THE COL-MAJOR CASE
  // TODO: Account for ldm; update this code if we change the corresponding, untested, store_matrix_sync
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
    auto pos = matrix_id * num_rows * num_cols + row * num_cols + col;
    p[pos] = a.x[i];
  }
}

__CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(__half* p, fragment<matrix_b, 8, 8, 4, __half, col_major> const & a, unsigned ldm) {
  asm("trap;");
  // NOTE: NOT LIKE THE COL-MAJOR CASE
  // TODO: Account for ldm; update this code if we change the corresponding, untested, load_matrix_sync
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
    auto pos = matrix_id * num_rows * num_cols + row * num_cols + col;
    p[pos] = a.x[i];
  }
}

*/

//
// Load functions for fragments of shape m16n8k8
//
__CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(__half* p, fragment<matrix_a, 16, 8, 8, __half, row_major> const & a, unsigned ldm) {
  // TODO: Account for ldm; update this code if we change the corresponding, untested, load_matrix_sync
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
      (row * num_cols + col) :
      (col * num_rows + row);
    p[pos] = a.x[i];
//      if ((double) a.x[i] != 0) printf("(%2d,%2d,%2d): After load acc; a.x[%3d] = %5.0f, (row,col) = (%3d,%3d) pos is %3d;\n",
//             threadIdx.x, threadIdx.y, threadIdx.z, i, (float) p[pos], row, col, (int) pos);
  }
}

__CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(__half* p, fragment<matrix_a, 16, 8, 8, __half, col_major> const & a, unsigned ldm) {
  // TODO: Account for ldm; update this code if we change the corresponding, untested, load_matrix_sync
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
      (row * num_cols + col) :
      (col * num_rows + row);
    p[pos] = a.x[i];
  }
}

__CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(__half* p, fragment<matrix_b, 16, 8, 8, __half, row_major> const & a, unsigned ldm) {
  // TODO: Account for ldm; update this code if we change the corresponding, untested, load_matrix_sync
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
      (row * num_cols + col) :
      (col * num_rows + row);
    p[pos] = a.x[i];
//      printf("(%2d,%2d,%2d) Load B row-major: a.x[%2d] = %5.0f, (row,col) = (%3d,%3d) pos is %3d;\n",
//             threadIdx.x, threadIdx.y, threadIdx.z, i, (float) p[pos], row, col, (int) pos);
  }
}

__CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(__half* p, fragment<matrix_b, 16, 8, 8, __half, col_major> const & a, unsigned ldm) {
  // TODO: Account for ldm; update this code if we change the corresponding, untested, load_matrix_sync
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
      (row * num_cols + col) :
      (col * num_rows + row);
    p[pos] = a.x[i];
//      printf("(%2d,%2d,%2d) Load B col-major: a.x[%2d] = %5.0f, (row,col) = (%3d,%3d) pos is %3d;\n",
//             threadIdx.x, threadIdx.y, threadIdx.z, i, (float) p[pos], row, col, (int) pos);
  }
}

/*
// UNTESTED!
__CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(__half* p, fragment<accumulator,16, 8, 8, __half> const & a, unsigned ldm, layout_t layout)
{
  // TODO: Account for ldm
  using fragment_type = decltype(detail::remove_ref_helper(a));
  bool is_row_major = (layout == wmma::mem_row_major);
  enum { M = 16, N = 8, K = 8 };
  enum { num_rows = M, num_cols = N };
  auto lane_id = detail::lane_id();

  auto groupID = lane_id >> 2;
  auto threadID_in_group = lane_id % 4;

  auto base_row = groupID;
  auto base_col = threadID_in_group * 2;

  #pragma unroll
  for(int i = 0; i < fragment_type::num_elements; i++) {
    auto row = base_row + ((i < 2) ? 0 : 8);
    auto col = base_col + (i & 0x1);
    auto pos = is_row_major ?
      row * num_cols + col :
      col * num_rows + row;
    p[pos] = a.x[i];
  }
}
*/


} // namespace wmma
} // namespace nvcuda

#undef __CUDA_IMMA__
#undef __CUDA_SUBBYTE_IMMA__
#undef __CUDA_MMA_DEVICE_DECL__
#undef __CUDA_AMPERE_MMA__

#endif /* !__CUDA_ARCH__ || __CUDA_ARCH__ >= 700 */

#endif /* __cplusplus && __CUDACC__ */


#endif // CUDA_MMA_SMALLER_HPP_

