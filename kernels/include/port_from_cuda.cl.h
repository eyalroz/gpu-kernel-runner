/**
 * @file port_from_cuda.cl.h
 *
 * @brief CUDA-flavor definitions for porting CUDA kernel code to OpenCL
 * will less changes required
 *
 * @note Changes you'll need to make on your own:
 *
 * - Replace `__local` with one of the aliases provided here
 * - Switch dynamic shared memory from being passed via a parameter
 *   to being available via an array variable of unspecified size.
 * - max(x,y) -> fmax
 *   (too risky to have a macro for this)
 *
 */
#ifndef PORT_FROM_CUDA_TO_OPENCL_CL_H_
#define PORT_FROM_CUDA_TO_OPENCL_CL_H_

#ifndef __CUDA_ARCH__

#define __shared__ __local
#define __global__ __kernel
#define __restrict__ restrict
#define __restrict restrict

// To keep your OpenCL kernel more immediately portable,
// avoid using __local directly, and instead use the following
// aliases. CUDA code can then decide whether these need to
// be replaced with  __shared__ or not.
//
#define __local_array __local
#define __local_ptr __local


#ifdef __CDT_PARSER__
#include "opencl_syntax_for_cdt_parser.cl.h"
#endif

typedef uchar  uint8_t;
typedef ushort uint16_t;
typedef uint   uint32_t;
typedef ulong  uint64_t;

inline float __frcp_rn(float x) { return native_recip(x); }
inline double __drcp_rn(double x) { return native_recip(x); }

/**
 * The following macro is intended to allow the same syntax for constructing compound types
 * in both OpenCL and CUDA. In CUDA, we would write float2 { foo, bar }; but in OpenCL we
 * would write that (float2) { foo, bar };
 */
#define make_compound(_compound_type) (_compound_type)

#endif // __CUDA_ARCH__

#endif // PORT_FROM_CUDA_TO_OPENCL_CL_H_
