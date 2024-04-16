/**
 * @file
 *
 * @copyright (c) 2024 Eyal Rozenberg <eyalroz1@gmx.com>.
 * @copyright (c) 2024 GE Healthcare.
 *
 * @note may require conditioning on Ampere-or-greater
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

#ifndef GKR_MMA_SMALLER_INTRINSICS_HPP_
#define GKR_MMA_SMALLER_INTRINSICS_HPP_

#include <mma.h> // This should, in particular, include crt/mma.hpp which we're extending here

#if defined(__cplusplus) && defined(__CUDACC__)


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define __CUDA_MMA_DEVICE_DECL__ static __device__ __inline__


namespace nvcuda {
namespace wmma {

// TODO: Do I need to use references to arrays rather than simply ptrs?
__CUDA_MMA_DEVICE_DECL__
void mma_sync_aligned_m16n8k16_row_col_f16_f16_f16_f16(
    half       d[4],
    half const a[8],
    half const b[4],
    half const c[4])
{
//    printf("(%2d,%2d,%2d): a = [ "
//           "%3.0f %3.0f %3.0f %3.0f "
//           "%3.0f %3.0f %3.0f %3.0f "
//           " ]\n",
//        threadIdx.x, threadIdx.y, threadIdx.z,
//        (float) a[0], (float) a[1], (float) a[2], (float) a[3],
//        (float) a[4], (float) a[5], (float) a[6], (float) a[7]
//        );

//    printf("(%2d,%2d,%2d): b = [ "
//           "%3.0f %3.0f %3.0f %3.0f "
//           " ]\n",
//        threadIdx.x, threadIdx.y, threadIdx.z,
//        (float) b[0], (float) b[1], (float) b[2], (float) b[3]
//        );

//    printf("(%2d,%2d,%2d): c = [ "
//           "%3.0f %3.0f %3.0f %3.0f "
//           " ]\n",
//        threadIdx.x, threadIdx.y, threadIdx.z,
//        (float) c[0], (float) c[1], (float) c[2], (float) c[3]
//        );

    unsigned const *A = reinterpret_cast<unsigned const *>(a);
    unsigned const *B = reinterpret_cast<unsigned const *>(b);
    unsigned const *C = reinterpret_cast<unsigned const *>(c);
    unsigned       *D = reinterpret_cast<unsigned       *>(d);
    asm(
        // mma.sync.aligned.m16n8k16.alayout.blayout.dtype.f16.f16.ctype
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, " // d
        "{%2,%3,%4,%5}, " // a
        "{%6,%7}, " // b
        "{%8,%9}" // c
        ";\n"
        : // outputs
            "=r"(D[0]), "=r"(D[1])
        : // inputs
            "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
            "r"(B[0]), "r"(B[1]),
            "r"(C[0]), "r"(C[1])
    );
}

/**
 *
 * When invoking this function, four sets of 8 lanes will each perform
 * an (8x4) x (4x8) matrix multplication: lenes 0-3,16-19 , and 8-sets
 * at offsets from these.
 *
 * The data layout for a row-major input matrix is: The i'th lane in
 * the set-of-8-lanes holds data from the i'th row of the matrix. As
 * K = 4, this data is four float values for each thread.
 *
 * For a column-major matrix, the lower lanes hold upper half-columns,
 * and the higher lanes hold lower half-columns, with every lane
 * holding data from exactly one column. Thus lanes 0 and 16 hold
 * together the column 0, with lane 0 holding its first 4 cells
 * and lane 16 hoding its last.
 *
 */
__CUDA_MMA_DEVICE_DECL__
void mma_sync_aligned_m8n8k4_row_col_f16_f16_f16_f16(
    half       d[8],
    half const a[4],
    half const b[4],
    half const c[8])
{
    unsigned const *A = reinterpret_cast<unsigned const *>(a);
    unsigned const *B = reinterpret_cast<unsigned const *>(b);
    unsigned const *C = reinterpret_cast<unsigned const *>(c);
    unsigned       *D = reinterpret_cast<unsigned       *>(d);
    asm(
        // .reg .f16x2 %Ra<2> %Rb<2> %Rc<4> %Rd<4>
        //mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16
        "mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16 "
        "{%0, %1, %2, %3}, " // d
        "{%4, %5}, " // a
        "{%6, %7}, " // b
        "{%8, %9, %10, %11}" // c
        ";\n"
        : // outputs
            "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : // inputs
            "r"(A[0]), "r"(A[1]),
            "r"(B[0]), "r"(B[1]),
            "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3])
    );
}

__CUDA_MMA_DEVICE_DECL__
void mma_sync_aligned_m16n8k8_row_col_f16_f16_f16_f16(
    half       d[4],
    half const a[4],
    half const b[2],
    half const c[4])
{
/*    if ((double)(a[0])>0 || (double)(a[1])>0 || (double)(a[2])>0 || (double)(a[3])>0 )
    printf("(%2d,%2d,%2d): half regs are: "
           "A 0,1,2,3: %4.f  %4.f  %4.f  %4.f | "
           "B 0,1: %4.f  %4.f | "
           "C 0,1,2,3: %4.f  %4.f  %4.f  %4.f | "
           "\n",
           threadIdx.x, threadIdx.y, threadIdx.z,
           (float) a[0], (float) a[1], (float) a[2], (float) a[3],
           (float) b[0], (float) b[1],
           (float) c[0], (float) c[1], (float) c[2], (float) c[3]
           );*/

    unsigned const *A = reinterpret_cast<unsigned const *>(a);
    unsigned const *B = reinterpret_cast<unsigned const *>(b);
    unsigned const *C = reinterpret_cast<unsigned const *>(c);
    unsigned       *D = reinterpret_cast<unsigned       *>(d);
/*    printf("In intrinsic; unsigned regs are: "
//           "D[0]: %8x D[1]: %8x "
           "A[0]: %8x A[1]: %8x "
           "B[0]: %8x "
           "C[0]: %8x C[1]: %8x "
           "\n",
//           D[0], D[1],
           A[0], A[1], B[0],  C[0], C[1]
           );*/
    asm(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0, %1}, " // d
        "{%2, %3}, " // a
        "{%4}, " //b
        "{%5, %6}" // c
        ";\n"
        : // outputs
            "=r"(D[0]), "=r"(D[1])
        : // inputs
            "r"(A[0]), "r"(A[1]),
            "r"(B[0]),
            "r"(C[0]), "r"(C[1])
    );
}

} // namespace wmma
} // namespace nvcuda

#undef __CUDA_MMA_DEVICE_DECL__

#endif /* !__CUDA_ARCH__ || __CUDA_ARCH__ >= 700 */

#endif /* __cplusplus && __CUDACC__ */

#endif   /* GKR_MMA_SMALLER_INTRINSICS_HPP_ */
