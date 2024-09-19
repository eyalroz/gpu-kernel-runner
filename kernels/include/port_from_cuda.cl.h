/**
 * @file port_from_cuda.cl.h
 *
 * @brief CUDA-flavor definitions for porting CUDA kernel code to OpenCL
 * with fewer changes required.
 *
 * @copyright (c) 2020-2024, GE HealthCare
 * @copyright (c) 2020-2024, Eyal Rozenberg
 *
 * @license BSD 3-clause license; see the `LICENSE` file or
 * @url https://opensource.org/licenses/BSD-3-Clause
 *
 * @note {Can be used for writing kernels targeting both CUDA and OpenCL
 * at once (alongside @ref port_from_opencl.cuh ).}
 *
 * @note Conventions you will need to follow:
 *
 *  | Instead of                | Use                                                | Explanation/Note                             |
 *  |:--------------------------|:---------------------------------------------------|:---------------------------------------------|
 *  | `__local` / `__shared`    | `__local_array` , `__local_var` or `__local_ptr` ) | Let a macro sort out thememory space marking |
 *  | `max(x,y)`                | `fmax(x,y)`                                        | it's too risky to define a `max(x,y)` macro  |
 *  | struct foo = { 12, 3.4 }; | struct foo = make_compound(foo){ 12, 3.4; }        | Allow for different construction syntax      |
 *  | constexpr                 | either CONSTEXPR_OR_CONSTANT_MEM, or an enum       |                                              |
 *
 * @note Use of dynamic shared memory is very different between OpenCL and CUDA, you'll
 * have to either avoid it or work the differences out yourself.
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

// constexpr in C++ is mostly-discretionary; in OpenCL C we'll just
// ignore it
#define constexpr

// This next definition is for "emulating" constexpr in OpenCL - or
// using the next closest thing - a global `__constant` memory space
// definition: The same syntax can be used in both OpenCL and CUDA,
// with CUDA actually producing `constexpr`, and OpenCL using `__constant`
#define CONSTEXPR_OR_CONSTANT_MEM __constant

#define static_assert(COND,MSG) typedef char static_assertion_at_line_##__LINE__[(COND)?1:-1]


#if defined(__CDT_PARSER__) || defined (__JETBRAINS_IDE__)
#include "opencl_syntax_for_ide_parser.cl.h"
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
#ifndef make_compound
#define make_compound(_compound_type) (_compound_type)
#endif

/**
 * OpenCL has (non-C-like) overloaded math primitives, while CUDA, ironically,
 * actually has many such primitives in C-style, not overloaded. Let's "implement"
 * some of them.
 *
 * @note
 * 1. Some functions are commented-out. They are available in CUDA, and
 *    apparently not available in OpenCL 3.0.
 * 2. no nanf
 * 3. For doubles, the CUDA and OpenCL function names are identical
 *
 * @{
 */

// Single-precision (i.e. FP32, float)

inline float acosf (float x) { return acos(x); }
inline float acoshf (float x) { return acosh(x); }
inline float asinf (float x) { return asin(x); }
inline float asinhf (float x) { return asinh(x); }
inline float atan2f (float y, float x) { return atan2(y, x); }
inline float atanf (float x) { return atan(x); }
inline float atanhf (float x) { return atanh(x); }
inline float cbrtf (float x) { return cbrt(x); }
inline float ceilf (float x) { return ceil(x); }
inline float copysignf (float x, float y) { return copysign(x, y); }
inline float cosf (float x) { return cos(x); }
inline float coshf (float x) { return cosh(x); }
inline float cospif (float x) { return cospi(x); }
// inline float cyl_bessel_i0f (float x) { return cyl_bessel_i0(x); }
// inline float cyl_bessel_i1f (float x) { return cyl_bessel_i1(x); }
inline float erfcf (float x) { return erfc(x); }
// inline float erfcinvf (float x) { return erfcinv(x); }
// inline float erfcxf (float x) { return erfcx(x); }
inline float erff (float x) { return erf (x); }
//inline float erfinvf (float x) { return erfinv(x); }
inline float exp10f (float x) { return exp10(x); }
inline float exp2f (float x) { return exp2(x); }
inline float expf (float x) { return exp(x); }
inline float expm1f (float x) { return expm1(x); }
inline float fabsf (float x) { return fabs(x); }
inline float fdimf (float x, float y) { return fdim(x, y); }
inline float floorf (float x) { return floor(x); }
inline float fmaf (float x, float y, float z) { return fma(x, y, z); }
inline float fmaxf (float x, float y) { return fmax(x, y); }
inline float fminf (float x, float y) { return fmin(x, y); }
inline float fmodf (float x, float y) { return fmod(x, y); }
inline float frexpf (float x, int* nptr) { return frexp(x, nptr); }
inline float hypotf (float x, float y) { return hypot(x, y); }
inline float ilogbf (float x) { return ilogb(x); }
//inline float j0f (float x) { return j0(x); }
//inline float j1f (float x) { return j1(x); }
//inline float jnf (int n, float x) { return jn(n, x); }
inline float ldexpf (float x, int exp) { return ldexp(x, exp); }
inline float lgammaf (float x) { return lgamma(x); }
//inline float llrintf (float x) { return llrint(x); }
//inline float llroundf (float x) { return llround(x); }
inline float log10f (float x) { return log10(x); }
inline float log1pf (float x) { return log1p(x); }
inline float log2f (float x) { return log2(x); }
inline float logbf (float x) { return logb(x); }
inline float logf (float x) { return log(x); }
//inline float lrintf (float x) { return lrint(x); }
//inline float lroundf (float x) { return lround(x); }
inline float modff (float x, float* iptr) { return modf (x, iptr); }
//inline float nearbyintf (float x) { return nearbyint(x); }
inline float nextafterf (float x, float y) { return nextafter(x, y); }
//inline float norm3df (float a, float b, float c) { return norm3d(a, b, c); }
//inline float norm4df (float a, float b, float c, float d) { return norm4d(a, b, c, d); }
//inline float normcdff (float x) { return normcdf (x); }
//inline float normcdfinvf (float x) { return normcdfinv(x); }
//inline float normf (int dim, const float* p) { return norm(dim, p); }
inline float powf (float x, float y) { return pow(x, y); }
//inline float rcbrtf (float x) { return rcbrt(x); }
inline float remainderf (float x, float y) { return remainder(x, y); }
inline float remquof (float x, float y, int* quo) { return remquo(x, y, quo); }
//inline float rhypotf (float x, float y) { return rhypot(x, y); }
inline float rintf (float x) { return rint(x); }
//inline float rnorm3df (float a, float b, float c) { return rnorm3d(a, b, c); }
//inline float rnorm4df (float a, float b, float c, float d) { return rnorm4d(a, b, c, d); }
//inline float rnormf (int dim, const float* p) { return rnorm(dim, p); }
inline float roundf (float x) { return round(x); }
inline float rsqrtf (float x) { return rsqrt(x); }
//inline float scalblnf (float x, long int n) { return scalbln(x, n); }
//inline float scalbnf (float x, int n) { return scalbn(x, n); }
//inline float sincosf (float x, float* sptr, float* cptr) { return sincos(x, sptr, cptr); }
//inline float sincospif (float x, float* sptr, float* cptr) { return sincospi(x, sptr, cptr); }
inline float sinf (float x) { return sin(x); }
inline float sinhf (float x) { return sinh(x); }
inline float sinpif (float x) { return sinpi(x); }
inline float sqrtf (float x) { return sqrt(x); }
inline float tanf (float x) { return tan(x); }
inline float tanhf (float x) { return tanh(x); }
inline float tgammaf (float x) { return tgamma(x); }
inline float truncf (float x) { return trunc(x); }
//inline float y0f (float x) { return y0(x); }
//inline float y1f (float x) { return y1(x); }
//inline float ynf (int n, float x) { return yn(n, x); }

// The following two functions are wrappers around plain division in OpenCL;
// they are necessary for CUDA compatibility, since in CUDA, plain division
// may be interpreted differently (e.g. uses a specific rounding mode) -
// depending on compilation options
// TODO: Make sure behavior doesn't change regardless of the compilation flags
inline float  fdividef (float  x, float  y) { return x / y; }
inline double fdivide  (double x, double y) { return x / y; }

// half-precision (i.e. FP16, half)

#ifdef PORT_FROM_CUDA_ENABLE_HALF_PRECISION
/**
 * Math function definitions for half-precision, rather than OpenCL's overridden
 * all-type functions
 */
inline half hceil(const half x) { ceil(h); }
inline half hcos(const half x) { cos(x); }
inline half hexp(const half x) { exp(x); }
inline half hexp10(const half x) { exp10(x); }
inline half hexp2(const half x) { exp2(x); }
inline half hfloor(const half x) { floor(x); }
inline half hlog(const half x) { log(x); }
inline half hlog10(const half x) { log10(x); }
inline half hlog2(const half x) { log3(x); }
// inline half hrcp(const half x) { rcp(x); }
inline half hrint(const half x) { rint(x); }
inline half hrsqrt(const half x) { rsqrt(x); }
inline half hsin(const half x) { sin(x); }
inline half hsqrt(const half x) { sqrt(x); }
inline half htrunc(const half x) { trunc(x); }
#endif // PORT_FROM_CUDA_ENABLE_HALF_PRECISION

#endif // __CUDA_ARCH__

#endif // PORT_FROM_CUDA_TO_OPENCL_CL_H_
