/**
 * @file
 *
 * @brief Conditionally-compiled definitions which let some IDE parsers -
 * currently JetBrains CLion and Eclipse CDT - better "accept" CUDA
 * sources without a specialized plugin.
 *
 * @copyright (c) 2020-2023, GE Healthcare.
 * @copyright (c) 2022-2023, Eyal Rozenberg.
 *
 * @license BSD 3-clause license; see the `LICENSE` file or
 * @url https://opensource.org/licenses/BSD-3-Clause
 *
 */
#ifndef CUDA_SYNTAX_FOR_IDE_PARSER_CUH_
#define CUDA_SYNTAX_FOR_IDE_PARSER_CUH_

#ifndef __OPENCL_VERSION__

#if defined(__CDT_PARSER__) || defined(__JETBRAINS_IDE__)
// These definitions will be ignored by the NVRTC compiler; they are only
// enabled for editing this file in a (non-CUDA-aware) IDE
template <typename T>
T max(const T& x, const T& y);

template <typename T>
T min(const T& x, const T& y);

void __syncthreads();

struct dim3 {
    int x, y, z;
};

dim3 threadIdx;
dim3 blockIdx;
dim3 blockDim;
dim3 gridDim;

#define __shared
#define __constant
#define __device__
#define __device_builtin__

inline float __fdividef(float __a, float __b);


#endif // defined(__CDT_PARSER__) || defined(__JETBRAINS_IDE__)

#endif // __OPENCL_VERSION__
#endif // CUDA_SYNTAX_FOR_IDE_PARSER_CUH_
