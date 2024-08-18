/**
 * @file
 *
 * @brief Conditionally-compiled definitions which let some IDE parsers -
 * currently JetBrains CLion and Eclipse CDT - better "accept" CUDA
 * sources without a specialized plugin.
 *
 * @copyright (c) 2020-2024, GE Healthcare.
 * @copyright (c) 2022-2024, Eyal Rozenberg.
 *
 * @license BSD 3-clause license; see the `LICENSE` file or
 * @url https://opensource.org/licenses/BSD-3-Clause
 *
 */
#ifndef CUDA_SYNTAX_FOR_IDE_PARSER_CUH_
#define CUDA_SYNTAX_FOR_IDE_PARSER_CUH_

#ifndef __OPENCL_VERSION__
#ifndef __CUDA_ARCH__

// These definitions will be ignored by the NVRTC compiler; they are only
// enabled for editing this file in a (non-CUDA-aware) IDE
#if defined(__CDT_PARSER__) || defined(__JETBRAINS_IDE__)

#include <vector_types.h>

template <typename T>
T max(const T& x, const T& y);

template <typename T>
T min(const T& x, const T& y);

void __syncthreads();

#ifndef __VECTOR_TYPES_H__
struct dim3 {
    int x, y, z;
};
#endif

//#ifndef CURAND_MTGP32_KERNEL_H
//dim3 threadIdx;
//dim3 blockDim;
//#endif
#ifndef __CUDA_BUILTIN_VAR
#define __CUDA_BUILTIN_VAR                                                     \
extern const __attribute__((device)) __attribute__((weak))
__CUDA_BUILTIN_VAR __cuda_builtin_threadIdx_t threadIdx;
__CUDA_BUILTIN_VAR __cuda_builtin_blockIdx_t blockIdx;
__CUDA_BUILTIN_VAR __cuda_builtin_blockDim_t blockDim;
__CUDA_BUILTIN_VAR __cuda_builtin_gridDim_t gridDim;
#endif

#ifndef __device__
#define __device__
#endif
#ifndef __restrict__
#define __restrict__
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __shared__
#define __shared__
#endif
#ifndef __constant__
#define __constant__
#endif
#ifndef __device_builtin__
#define __device_builtin__
#endif

#ifndef __DEVICE__
#define __DEVICE__
#endif

inline float __fdividef(float __a, float __b);
inline float floorf(float __f);

// These definitions are only correct w.r.t. size. Don't rely on them overmuch
struct __nv_bfloat16 { float x; };
struct __nv_bfloat162 { float x; };
struct __nv_bfloat164 { float x,y; };

#endif // defined(__CDT_PARSER__) || defined(__JETBRAINS_IDE__)

#endif // __CUDA_ARCH__
#endif // __OPENCL_VERSION__
#endif // CUDA_SYNTAX_FOR_IDE_PARSER_CUH_
