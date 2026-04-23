/**
 * @file
 *
 * @brief Definitions and aliases of functions which are built-in for OpenCL,
 * but not for CUDA. These are mostly functions from §6.15 of the OpenCL
 * standard, as well as conversion 'operators' from §6.4.3 and §6.4.4.2 .
 * Variants of functions working on vector types (intN, floatN etc.) do
 * _not_ appear in this file; find them in @ref vectorized/builtin_functions.cuh
 *
 * @copyright (c) 2020-2026, GE HealthCare
 * @copyright (c) 2020-2026, Eyal Rozenberg
 *
 * @license BSD 3-clause license; see the `LICENSE` file or
 * @url https://opensource.org/licenses/BSD-3-Clause
 *
 * @note
 *
 */
#ifndef PORT_FROM_OPENCL_BUILTIN_FUNCTIONS_CUH_
#define PORT_FROM_OPENCL_BUILTIN_FUNCTIONS_CUH_

#ifndef __OPENCL_VERSION__

#include "opencl_scalar_types.cuh"
#include "opencl_defines.cuh"

#include <vector_types.h>
#include <cuda_fp16.h>

#if defined(__CDT_PARSER__) || defined (__JETBRAINS_IDE__)
#endif

// // The CLion clang-based compiler apparently includes device_functions.h in a way which
// extern "C" {
// int                    __float2int_rz(float x);
// int                    __float2int_ru(float x);
// int                    __float2int_rn(float x);
// int                    __float2int_rd(float x);
//
// int                    __double2int_rz(double x);
// int                    __double2int_ru(double x);
// int                    __double2int_rn(double x);
// int                    __double2int_rd(double x);
//
// unsigned long long     __float2ull_rz(float x);
// unsigned long long     __float2ull_ru(float x);
// unsigned long long     __float2ull_rn(float x);
// unsigned long long     __float2ull_rd(float x);
//
// long long              __float2ll_rz(float x);
// long long              __float2ll_ru(float x);
// long long              __float2ll_rn(float x);
// long long              __float2ll_rd(float x);
//
// uint                   __float2uint_rz(float x);
// uint                   __float2uint_ru(float x);
// uint                   __float2uint_rn(float x);
// uint                   __float2uint_rd(float x);
//
// uint                   __double2uint_rz(double x);
// uint                   __double2uint_ru(double x);
// uint                   __double2uint_rn(double x);
// uint                   __double2uint_rd(double x);
//
// unsigned long long     __double2ull_rz(double x);
// unsigned long long     __double2ull_ru(double x);
// unsigned long long     __double2ull_rn(double x);
// unsigned long long     __double2ull_rd(double x);
//
// long long              __double2ll_rz(double x);
// long long              __double2ll_ru(double x);
// long long              __double2ll_rn(double x);
// long long              __double2ll_rd(double x);
//
//
// int __half2int_rn(const __half h);
// int __half2int_rd(const __half h);
// int __half2int_ru(const __half h);
// short int __half2short_rn(const __half h);
// short int __half2short_rd(const __half h);
// short int __half2short_ru(const __half h);
// unsigned int __half2uint_rn(const __half h);
// unsigned int __half2uint_rd(const __half h);
// unsigned int __half2uint_ru(const __half h);
// unsigned short int __half2ushort_rn(const __half h);
// unsigned short int __half2ushort_rd(const __half h);
// unsigned short int __half2ushort_ru(const __half h);
// unsigned long long int __half2ull_rn(const __half h);
// unsigned long long int __half2ull_rd(const __half h);
// unsigned long long int __half2ull_ru(const __half h);
// long long int __half2ll_rn(const __half h);
// long long int __half2ll_rd(const __half h);
// long long int __half2ll_ru(const __half h);
// }


#endif // defined(__CDT_PARSER__) || defined (__JETBRAINS_IDE__)


// §6.4.3 Explicit conversions
// ===========================

/**
 * The form of explicit conversion functions for scalar types is:
 *
 *   destType convert_destType<_sat><_roundingMode>(sourceType)
 *
 * ... and the basic types to/from we can convert are:
 * char, uchar, short, ushort, int, uint, long, ulong, float
 *
 * To avoid massive repetitions, we'll be using macros and templates to define
 * the different conversion functions
 */

// Note: The character '2' in these definitions does not describe a 2-element vector,
// but is rather a shorthand for the word 'to', e.g. "float to int" becomes `__float2int`.
#define PORT_FROM_OPENCL_DEFINE_FLOAT_TO_INT_CONVERSIONS(src, dst, dst_suffix) \
inline __device__ dst convert_ ## dst ## _rtz (src v) { return static_cast<dst>(__ ## src ## 2 ## dst_suffix ## _rz (v)); } \
inline __device__ dst convert_ ## dst ## _rte (src v) { return static_cast<dst>(__ ## src ## 2 ## dst_suffix ## _rn (v)); } \
inline __device__ dst convert_ ## dst ## _rtp (src v) { return static_cast<dst>(__ ## src ## 2 ## dst_suffix ## _ru (v)); } \
inline __device__ dst convert_ ## dst ## _rtn (src v) { return static_cast<dst>(__ ## src ## 2 ## dst_suffix ## _rd (v)); } \
inline __device__ dst convert_ ## dst         (src v) { return convert_ ## dst ## _rtz (v); }

#define PORT_FROM_OPENCL_DEFINE_CONVERSIONS_TO_INT_TYPE(dst, dst_suffix, dst_suffix_for_half) \
template <typename src> \
inline __device__ dst convert_ ## dst (src v) noexcept { return static_cast<dst>(v); } \
PORT_FROM_OPENCL_DEFINE_FLOAT_TO_INT_CONVERSIONS(half, dst, dst_suffix_for_half) \
PORT_FROM_OPENCL_DEFINE_FLOAT_TO_INT_CONVERSIONS(float, dst, dst_suffix) \
PORT_FROM_OPENCL_DEFINE_FLOAT_TO_INT_CONVERSIONS(double, dst, dst_suffix)

// TODO: Define _sat conversions
// TODO: Define conversions to float, e.g. double2float

PORT_FROM_OPENCL_DEFINE_CONVERSIONS_TO_INT_TYPE(int, int, int)
PORT_FROM_OPENCL_DEFINE_CONVERSIONS_TO_INT_TYPE(char, int, short)
PORT_FROM_OPENCL_DEFINE_CONVERSIONS_TO_INT_TYPE(uchar, uint, ushort)
PORT_FROM_OPENCL_DEFINE_CONVERSIONS_TO_INT_TYPE(short, int, short)
PORT_FROM_OPENCL_DEFINE_CONVERSIONS_TO_INT_TYPE(ushort, uint, ushort)
PORT_FROM_OPENCL_DEFINE_CONVERSIONS_TO_INT_TYPE(long, ll, ll)
PORT_FROM_OPENCL_DEFINE_CONVERSIONS_TO_INT_TYPE(ulong, ull, ull)
// Note: There is no 'long long' in OpenCL


// §6.4.4.2. Reinterpreting Types Using as_type()
// =============================================================

// "All data types described in Built-in Scalar Data Types and
// Built-in Vector Data Types (except bool, void, and half [19])
// may be also reinterpreted as another data type of the same size"

#define PORT_FROM_OPENCL_DEFINE_ASTYPE(dst) \
template <typename src> \
inline dst as_ ## dst (src v) \
{ \
    static_assert(sizeof(dst) == sizeof(src), "as_type for types of different size is not supported"); \
    return reinterpret_cast<dst>(v); \
}

PORT_FROM_OPENCL_DEFINE_ASTYPE(char)
PORT_FROM_OPENCL_DEFINE_ASTYPE(uchar)
PORT_FROM_OPENCL_DEFINE_ASTYPE(short)
PORT_FROM_OPENCL_DEFINE_ASTYPE(ushort)
PORT_FROM_OPENCL_DEFINE_ASTYPE(int)
PORT_FROM_OPENCL_DEFINE_ASTYPE(uint)
PORT_FROM_OPENCL_DEFINE_ASTYPE(long)
PORT_FROM_OPENCL_DEFINE_ASTYPE(ulong)
// half is _not_ supported for as_type
PORT_FROM_OPENCL_DEFINE_ASTYPE(float)
PORT_FROM_OPENCL_DEFINE_ASTYPE(double)

// §6.15.1. Work-Item Functions
// ============================

namespace detail_ {

constexpr __host__ __device__ inline unsigned int
get_dim3_element(const dim3& d3, int index, unsigned fallback) noexcept
{
    switch (index) {
        case 0:  return d3.x;
        case 1:  return d3.y;
        case 2:  return d3.z;
        default: return fallback;
    }
}

} // namespace detail_


/// Returns the number of dimensions in use. This is the value given to the work_dim argument specified in clEnqueueNDRangeKernel.
inline uint get_work_dim() noexcept { return 3; }

/// Returns the unique global work-item ID value for dimension identified by dimension_index. The global work-item ID specifies the work-item ID based on the number of global work-items specified to execute the kernel.
/// Valid values of dimension_index are 0 to get_work_dim() - 1. For other values of dimension_index, get_global_id() returns 0.
inline size_t get_global_id(int dimension_index) noexcept
{
    // Note: We could have used:
    //
    //  return detail_::get_dim3_element(threadIdx, dimension_index) +
    //  detail_::get_dim3_element(blockIdx, dimension_index) *
    //  detail_::get_dim3_element(blockDim, dimension_index) noexcept;
    //
    // But I'm not sure whether we can trust the compiler to optimize
    // all of that away

    switch (dimension_index) {
        case 0:  return threadIdx.x + static_cast<size_t>(blockIdx.x) * blockDim.x;
        case 1:  return threadIdx.y + static_cast<size_t>(blockIdx.y) * blockDim.y;
        case 2:  return threadIdx.z + static_cast<size_t>(blockIdx.z) * blockDim.z;
        default: return 0;
    }
}

/**
 * Returns the number of local work-items specified in dimension identified by dimension_index.
 *
 * @note in OpenCL per se, the logic here is more complex, as one may enqueue OpenCL kernels
 * without specifying the local work size, and the OpenCL implementation chooses one for
 * the kernel. OpenCL may also support a non-uniform work-group size. Both of these are
 * not supported In CUDA, this is not supported.
 *

 * Valid values of dimension_index are 0 to get_work_dim() - 1. For other values of dimension_index, get_local_size() returns 1.
 */
inline size_t get_local_size(unsigned dimension_index) noexcept
{
    enum { Fallback = 0 };
    return detail_::get_dim3_element(blockDim, dimension_index, 1);
}

/**
 * Returns the same value as that returned by `get_local_size(dimension_index)` if the kernel is executed with
 * a uniform work-group size - which, in CUDA kernels, is _always_ the case.
 *
 * @note In non-NVIDIA-GPU OpenCL implementations, this function is not redundant and has non-trivial
 * logic.
 *
 * @param dimension_index[in] A value in the range `0`..`get_work_dim() - 1`.
 *
 * @return If the value of @p dimension_index is outside the vlaid range, 1 is returned.
 */
inline size_t get_enqueued_local_size(uint dimension_index) noexcept { return get_local_size(dimension_index); }

/// Returns the unique local work-item ID, i.e. a work-item within a specific work-group for dimension identified by dimension_index.
/// Valid values of dimension_index are 0 to get_work_dim() - 1. For other values of dimension_index, get_local_id() returns 0.
inline size_t get_local_id(int dimension_index) noexcept
{
    enum { Fallback = 0 };
    return detail_::get_dim3_element(threadIdx, dimension_index, Fallback);
}

/// Returns the number of work-groups that will execute a kernel for dimension identified by dimension_index.
/// Valid values of dimension_index are 0 to get_work_dim() - 1. For other values of dimension_index, get_num_groups() returns 1.
inline size_t get_num_groups(uint dimension_index) noexcept
{
    enum { Fallback = 1 };
    return detail_::get_dim3_element(gridDim, dimension_index, Fallback);
}

/// Returns the number of global work-items specified for dimension identified by dimension_index. This value is given by the global_work_size argument to clEnqueueNDRangeKernel.
/// Valid values of dimension_index are 0 to get_work_dim() - 1. For other values of dimension_index, get_global_size() returns 1.
inline size_t get_global_size(uint dimension_index) noexcept
{
    return static_cast<size_t>(get_num_groups(dimension_index)) * get_local_size(dimension_index);
}

/// get_group_id returns the work-group ID which is a number from 0 .. get_num_groups(dimension_index) - 1.
/// Valid values of dimension_index are 0 to get_work_dim() - 1. For other values, get_group_id() returns 0.
inline size_t get_group_id(int dimension_index) noexcept
{
    enum { Fallback = 0 };
    return detail_::get_dim3_element(blockIdx, dimension_index, 0);
}

/**
 * get_global_offset returns the offset values for 3D global thread coordinates in the
 * launch grid - which, for CUDA kernels, is always 0, since such offsets are not supported.
 *
 * @note Requires support for OpenCL C 1.1 or newer.
 *
 * @param dimension_index[in] A value in the range `0`..`get_work_dim() - 1`.
 */
inline size_t get_global_offset(uint dimension_index) noexcept { (void) dimension_index; return 0; }

/**
 * Returns the work-item's 1-dimensional global ID.
 *
 * Since in CUDA, all grids are 3-dimensional and have no offset, it is a rather
 * straightforward formula.
 */
inline size_t get_global_linear_id() noexcept
{
    return (get_global_id(2) * get_global_size(1) * get_global_size(0))
         + (get_global_id(1) * get_global_size(0))
         +  get_global_id(0);
}

/**
 * Returns the work-items 1-dimensional local ID.
 *
 * Since in CUDA, all blocks are 3-dimensional and have no offset, it is a rather
 * straightforward formula.
*/
inline size_t get_local_linear_id() noexcept
{
    return (get_local_id(2) * get_local_size(1) * get_local_size(0))
         + (get_local_id(1) * get_local_size(0))
         +  get_local_id(0);
}

// Not implemented: functions for subgroups

// §6.15.2. Math Functions
// =======================

// With math functions we are in somewhat of a pickle, since CUDA exposes
// double-precision functions having the same name as OpenCL's gentype
// functions. The best we can do is expand the overload sets with the
// different-precision variants.

// TODO: What about the implicit inclusions of __clang_cuda_math.h and
//  __clang_cuda_runtime_wrapper.h by the clang parser? :-(

namespace detail_ {

typedef union {
    half value;
    struct {
        unsigned int significand : 10;
        unsigned int exponent    : 5;
        unsigned int sign        : 1;
    } parts;
} destructured_half;

typedef union {
    float value;
    struct {
        unsigned int significand : 23;
        unsigned int exponent    : 8;
        unsigned int sign        : 1;
    } parts;
} destructured_float;

typedef union {
    double value;
    struct {
        long unsigned int significand : 53;
             unsigned int exponent    : 11;
             unsigned int sign        : 1;
    } parts;
} destructured_double;

} // namespace detail_

// double-precision parameter, but missing in CUDA
// ------------------------------------------------

inline double acospi(double x) noexcept { return acos(x) * M_1_PI; }
inline double asinpi(double x) noexcept { return asin(x) * M_1_PI; }
// sincos is already defined in cuda
inline double atan2pi(double y, double x) noexcept { return atan2(y, x) * M_1_PI; }
inline double tanpi(double x) noexcept { return tan(M_PI * x); }

inline double fract(double x, double *iptr) noexcept
{
    double floor_ = floor(x);
    *iptr = floor_;
    double highest_fractional_under_1 = 0x1.ffffffffffffep-1;
    // ... which is the normal form of 0x0.ffffffffffff - 52 bits set
    return fmin(x - floor_, highest_fractional_under_1);
}

inline double frexp(double x, int *exp) noexcept
{
    detail_::destructured_double x_;
    x_.value = x;
    *exp = x_.parts.exponent;
    return x_.parts.significand;
}

inline double lgamma_r(double x, int *signp) noexcept
{
    detail_::destructured_float gamma_;
    gamma_.value = tgammaf(x);
    *signp = gamma_.parts.sign;
    gamma_.parts.sign = 0; // make it positive;
    return logf(gamma_.value);
}

inline int ilogb(double x) noexcept { return reinterpret_cast<detail_::destructured_double&>(x).parts.exponent; }

template <typename T, typename S>
inline double maxmag(T x, S y) noexcept
{
    double abs_x = fabs(x);
    double abs_y = fabs(y);
    if (abs_x == abs_y) { return fmax(x,y); }
    return (abs_x > abs_y) ? x : y;
}

inline double mad(double a, double b, double c) noexcept { return fma(a, b, c); }

// float parameter
// ---------------

inline float acos(float x) noexcept { return acosf(x); }
inline float acosh(float x) noexcept { return acoshf(x); }
inline float acospi(float x) noexcept { return acosf(x) * M_1_PI_F; }
inline float asin(float x) noexcept { return asinf(x); }
inline float asinh(float x) noexcept { return asinhf(x); }
inline float asinpi(float x) noexcept { return asinf(x) * M_1_PI_F; }
inline float atan(float y_over_x) noexcept { return atanf(y_over_x); }
inline float atan2(float y, float x) noexcept { return atan2f(y, x); }
inline float atanh(float x) noexcept { return atanhf(x); }
inline float atanpi(float x) noexcept { return atanf(x) * M_1_PI_F; }
inline float atan2pi(float y, float x) noexcept { return atan2f(y, x) * M_1_PI_F; }
inline float cbrt(float x) noexcept { return cbrtf(x); }
inline float ceil(float x) noexcept { return ceilf(x); }
inline float copysign(float x, float y) noexcept { return copysignf(x, y); }
inline float cos(float x) noexcept { return cosf(x); }
inline float cosh(float x) noexcept { return coshf(x); }
inline float erfc(float x) noexcept { return erfcf(x); }
inline float erf(float x) noexcept { return erff(x); }
inline float exp(float x) noexcept { return expf(x); }
inline float exp2(float x) noexcept { return exp2f(x); }
inline float exp10(float x) noexcept { return exp10f(x); }
inline float expm1(float x) noexcept { return expm1f(x); }
inline float fabs(float x) noexcept { return fabsf(x); }
inline float fdim(float x, float y) noexcept { return fdimf(x, y); }
inline float floor(float x) noexcept { return floorf(x); }
inline float fma(float a, float b, float c) { return fmaf(a, b, c); }
inline float fmax(float x, float y) noexcept { return fmaxf(x, y); }
// Not implemented: gentyped fmax(gentyped x, float y)
inline float fmin(float x, float y) noexcept { return fminf(x, y); }
// Not implemented: gentyped fmin(gentyped x, float y)
inline float fmod(float x, float y) noexcept { return fmodf(x, y); }
inline float fract(float x, float *iptr) noexcept
{
    float floor_ = floor(x);
    *iptr = floor_;
    float highest_fractional_under_1 = 0x1.fffffep-1f;
    // ... which is the normal form of 0x0.ffffff - 24 bits set
    return fmin(x - floor_, highest_fractional_under_1);
}

inline float frexp(float x, int *exp) noexcept
{
    detail_::destructured_float x_;
    x_.value = x;
    *exp = x_.parts.exponent;
    return x_.parts.significand;
}

inline float hypot(float x, float y) noexcept { return hypotf(x, y); }
inline int ilogb(float x) noexcept { return reinterpret_cast<detail_::destructured_float&>(x).parts.exponent; }
inline float ldexp(float x, int k) noexcept { return ldexpf(x, k); }
inline float lgamma(float x) noexcept { return lgammaf(x); }
inline float lgamma_r(float x, int *signp) noexcept
{
    detail_::destructured_float gamma_;
    gamma_.value = tgammaf(x);
    *signp = gamma_.parts.sign;
    gamma_.parts.sign = 0; // make it positive;
    return logf(gamma_.value);
}
inline float log(float x) noexcept { return logf(x); }
inline float log2(float x) noexcept { return log2f(x); }
inline float log10(float x) noexcept { return log10f(x); }
inline float log1p(float x) noexcept { return log1pf(x); }
inline float logb(float x) noexcept { return logbf(x); }
inline float mad(float a, float b, float c) noexcept { return fmaf(a, b, c); }
inline float maxmag(float x, float y) noexcept
{
    float abs_x = fabsf(x);
    float abs_y = fabsf(y);
    if (abs_x == abs_y) { return fmaxf(x,y); }
    return (abs_x > abs_y) ? x : y;
}
inline float minmag(float x, float y) noexcept
{
    float abs_x = fabsf(x);
    float abs_y = fabsf(y);
    if (abs_x == abs_y) { return fminf(x,y); }
    return (abs_x < abs_y) ? x : y;
}
inline float modf(float x, float *iptr) noexcept { return modff(x, iptr); }
// Not implementing nan, since it doesn't take a parameter which could distinguish float's from doubles etc.
inline float nextafter(float x, float y) noexcept { return nextafterf(x, y); }
inline float pow(float x, float y) noexcept { return powf(x, y); }
inline float powr(float x, float y) noexcept { return  exp2f(y * log2f(x)); }
inline float remainder(float x, float y) noexcept { return remainderf(x, y); }
inline float remquo(float x, float y, int *quo) noexcept { return remquof(x, y, quo); }
inline float rint(float x) noexcept { return rintf(x); }
// No root, rootn function in CUDA.
inline float round(float x) noexcept { return roundf(x); }
inline float sin(float x) noexcept { return sinf(x); }
inline float sincos(float x, float *cosval) { float sinval; sincosf(x, &sinval, cosval); return sinval; }
inline float sinh(float x) noexcept { return sinhf(x); }
inline float sqrt(float x) noexcept { return sqrtf(x); }
inline float tan(float x) noexcept { return tanf(x); }
inline float tanh(float x) noexcept { return tanhf(x); }
inline float tanpi(float x) noexcept { return tanf(M_PI_F * x); }
inline float tgamma(float x) noexcept { return tgammaf(x); }
inline float trunc(float x) noexcept { return truncf(x); }

#ifndef __CLANG_CUDA_RUNTIME_WRAPPER_H__
inline float rsqrt(float x) { return rsqrtf(x); }
inline float sinpi(float x) { return sinpif(x); }
inline float cospi(float x) { return cospif(x); }
#endif

// half-precision parameter
// -------------------------

// Note: CUDA (as of 13.1) does not provide half-precision versions of most of the
// math functions it offers for float- and double-precision. Also, some OpenCL functions
// which, for double and float, we have implemented using Pi-related constants -
// but we don't have appropriate definitions for the half-precision versions of these
// therefore, many of
// the "implementations" here are commented-out.

//inline half acos(half x) noexcept { return hacos(x); }
//inline half acosh(half x) noexcept { return hacosh(x); }
//inline half acospi(half x) noexcept { return hacos(x) * (half) M_1_PI_F; }
//inline half asin(half x) noexcept { return hasin(x); }
//inline half asinh(half x) noexcept { return hasinh(x); }
//inline half asinpi(half x) noexcept { return hasin(x) * (half) M_1_PI_F; }
//inline half atan(half y_over_x) noexcept { return hatan(y_over_x); }
//inline half atan2(half y, half x) noexcept { return hatan2(y, x); }
//inline half atanh(half x) noexcept { return hatanh(x); }
//inline half atanpi(half x) noexcept { return hatan(x) * M_1_PI_F; }
//inline half atan2pi(half y, half x) noexcept { return hatan2(y, x) * (half) M_1_PI_F; }
//inline half cbrt(half x) noexcept { return hcbrt(x); }
inline half ceil(half x) noexcept { return hceil(x); }
//inline half copysign(half x, half y) noexcept { return hcopysign(x, y); }
inline half cos(half x) noexcept { return hcos(x); }
//inline half cosh(half x) noexcept { return hcosh(x); }
//inline half cospi(half x) noexcept { return hcospi(x); }
//inline half erfc(half x) noexcept { return herfc(x); }
//inline half her(half x) noexcept { return herf(x); }
inline half exp(half x) noexcept { return hexp(x); }
inline half exp2(half x) noexcept { return hexp2(x); }
inline half exp10(half x) noexcept { return hexp10(x); }
//inline half expm1(half x) noexcept { return hexpm1(x); }
//inline half fabs(half x) noexcept { return hfabs(x); }
//inline half fdim(half x, half y) noexcept { return hfdim(x, y); }
inline half floor(half x) noexcept { return hfloor(x); }
//inline half fma(half a, half b, half c) { return hfma(a, b, c); }
//inline half fmax(half x, half y) noexcept { return hfmax(x, y); }
// Not implemented: gentyped fmax(gentyped x, half y)
//inline half fmin(half x, half y) noexcept { return hfmin(x, y); }
// Not implemented: gentyped fmin(gentyped x, half y)
//inline half fmod(half x, half y) noexcept { return hfmod(x, y); }
inline half fract(half x, half *iptr) noexcept
{
    half floor_ = floor(x);
    *iptr = floor_;
    half highest_fractional_under_1 = (half) 0x1.7ffep-1;
    // ... which is the normal form of 0x0.ffffff - 24 bits set
    return fmin(x - floor_, highest_fractional_under_1);
}

inline half frexp(half x, int *exp) noexcept
{
    detail_::destructured_half x_;
    x_.value = x;
    *exp = x_.parts.exponent;
    return x_.parts.significand;
}

inline half hypot(half x, half y) noexcept { return hypotf(x, y); }
inline int ilogb(half x) noexcept { return reinterpret_cast<detail_::destructured_half&>(x).parts.exponent; }
//inline half ldexp(half x, int k) noexcept { return hldexp(x, k); }
// TODO: implement ldexp with vectorized and non-vectorized k for halfn with n = 3,8,16
//inline half lgamma(half x) noexcept { return hlgamma(x); }
//inline half lgamma_r(half x, int *signp) noexcept
//{
//    detail_::destructured_half gamma_;
//    gamma_.value = htgamma(x);
//    *signp = gamma_.parts.sign;
//    gamma_.parts.sign = 0; // make it positive;
//    return hlog(gamma_.value);
//}
inline half log(half x) noexcept { return hlog(x); }
inline half log2(half x) noexcept { return hlog2(x); }
inline half log10(half x) noexcept { return hlog10(x); }
//inline half log1p(half x) noexcept { return hlog1p(x); }
//inline half logb(half x) noexcept { return hlogb(x); }
//inline half mad(half a, half b, half c) noexcept { return hfma(a, b, c); }
//inline half maxmag(half x, half y) noexcept
//{
//    half abs_x = hfabs(x);
//    half abs_y = hfabs(y);
//    if (abs_x == abs_y) { return hfmax(x,y); }
//    return (abs_x > abs_y) ? x : y;
//}
//inline half minmag(half x, half y) noexcept
//{
//    half abs_x = hfabs(x);
//    half abs_y = hfabs(y);
//    if (abs_x == abs_y) { return hfmin(x,y); }
//    return (abs_x < abs_y) ? x : y;
//}

// inline half modf(half x, half *iptr) noexcept { return hmodf(x, iptr); }
// Not implementing nan(), since it doesn't take a parameter which could distinguish half's from doubles etc.
// inline half nextafter(half x, half y) noexcept { return hnextafter(x, y); }
// inline half pow(half x, half y) noexcept { return hpow(x, y); }
//inline half powr(half x, half y) noexcept { return hpowr(x, y); }
// inline half remainder(half x, half y) noexcept { return hremainder(x, y); }
// inline half remquo(half x, half y, int *quo) noexcept { return hremquo(x, y, *quo); }
inline half rint(half x) noexcept { return hrint(x); }
// No root, rootn function in CUDA.
// inline half round(half x) noexcept { return hround(x); }
inline half sin(half x) noexcept { return hsin(x); }
// inline half sincos(half x, half *cosval) { half sinval; hsincos(x, &sinval, &cosval); return *cosval; }
// inline half sinh(half x) noexcept { return hsinh(x); }
inline half sqrt(half x) noexcept { return hsqrt(x); }
// inline half tan(half x) noexcept { return htan(x); }
// inline half tanh(half x) noexcept { return htanh(x); }
// inline half tanpi(half x) noexcept { return htan((half) M_PI_F * x); }
// inline half tgamma(half x) noexcept { return htgamma(x); }
inline half trunc(half x) noexcept { return htrunc(x); }

#ifndef __CLANG_CUDA_RUNTIME_WRAPPER_H__
inline half rsqrt(half x) { return hrsqrt(x); }
// inline half sinpi(half x) { return hsinpi(x); }
// inline half cospi(half x) { return hcospi(x); }
#endif

// `half_` functions from Table 11
// ---------------

// TODO: add these

// `native_` functions from Table 12
// ---------------------------------

// TODO: add these


// §6.15.3. Integer Functions
// --------------------------------------------------------------

// Some of the OpenCL integer-math functions have CUDA equivalents with the exact same
// names - either with an identical signature or through implicit conversions. These are:
// min, max .

// These functions should be implemented for each of:
//
// char, uchar, short, ushort, int, uint, long, ulong
// and for their vector types

// char:
inline uchar abs(char x) { return x >= 0 ? x : -x; }
inline uchar abs_diff(char x, char y) { return x < y ? (y - x) : (x - y); }
char add_sat(char x, char y);
inline char hadd(char x, char y) { return (x >> 1) + (y >> 1) + (x & y & 1); }
char rhadd(char x, char y);
inline char clamp(char x, char minval, char maxval) { return min(max(x, minval), maxval); }
char clz(char x);
char ctz(char x);
char mad_hi(char a, char b, char c);
char mad_sat(char a, char b, char c);
char mul_hi(char x, char y);
char rotate(char v, char i);
char sub_sat(char x, char y);
char popcount(char x);
short upsample(char hi, uchar lo);

// uchar:
inline uchar abs(uchar x) { return x; }
inline uchar abs_diff(uchar x, uchar y) { return x < y ? (y - x) : (x - y); }
uchar add_sat(uchar x, uchar y);
inline uchar hadd(uchar x, uchar y) { return (x >> 1) + (y >> 1) + (x & y & 1); }
uchar rhadd(uchar x, uchar y);
inline uchar clamp(uchar x, uchar minval, uchar maxval) { return min(max(x, minval), maxval); }
uchar clamp(uchar x, char minval, char maxval);
uchar clz(uchar x);
uchar ctz(uchar x);
uchar mad_hi(uchar a, uchar b, uchar c);
uchar mad_sat(uchar a, uchar b, uchar c);
uchar mul_hi(uchar x, uchar y);
uchar rotate(uchar v, uchar i);
uchar sub_sat(uchar x, uchar y);
uchar popcount(uchar x);
ushort upsample(uchar hi, uchar lo);

// short:
inline ushort abs(short x) { return x >= 0 ? x : -x; }
inline ushort abs_diff(short x, short y) { return x < y ? (y - x) : (x - y); }
short add_sat(short x, short y);
inline short hadd(short x, short y) { return (x >> 1) + (y >> 1) + (x & y & 1); }
short rhadd(short x, short y);
inline short clamp(short x, short minval, short maxval) { return min(max(x, minval), maxval); }
short clz(short x);
short ctz(short x);
short mad_hi(short a, short b, short c);
short mad_sat(short a, short b, short c);
short mul_hi(short x, short y);
short rotate(short v, short i);
short sub_sat(short x, short y);
short popcount(short x);
int upsample(short hi, ushort lo);

// ushort:
inline ushort abs(ushort x) { return x; }
inline ushort abs_diff(ushort x, ushort y) { return x < y ? (y - x) : (x - y); }
ushort add_sat(ushort x, ushort y);
inline ushort hadd(ushort x, ushort y) { return (x >> 1) + (y >> 1) + (x & y & 1); }
ushort rhadd(ushort x, ushort y);
inline ushort clamp(ushort x, ushort minval, ushort maxval) { return min(max(x, minval), maxval); }
ushort clamp(ushort x, short minval, short maxval);
ushort clz(ushort x);
ushort ctz(ushort x);
ushort mad_hi(ushort a, ushort b, ushort c);
ushort mad_sat(ushort a, ushort b, ushort c);
ushort mul_hi(ushort x, ushort y);
ushort rotate(ushort v, ushort i);
ushort sub_sat(ushort x, ushort y);
ushort popcount(ushort x);
uint upsample(ushort hi, ushort lo);

// int:
inline uint abs(int x) { return x >= 0 ? x : -x; }
inline uint abs_diff(int x, int y) { return x < y ? (y - x) : (x - y); }
int add_sat(int x, int y);
inline int hadd(int x, int y) { return (x >> 1) + (y >> 1) + (x & y & 1); }
int rhadd(int x, int y);
inline int clamp(int x, int minval, int maxval) { return min(max(x, minval), maxval); }
int clz(int x);
int ctz(int x);
int mad_hi(int a, int b, int c);
int mad_sat(int a, int b, int c);
int mul_hi(int x, int y);
int rotate(int v, int i);
int sub_sat(int x, int y);
inline int popcount(int x) { return __popc(reinterpret_cast<uint&>(x)); }
long upsample(int hi, uint lo);

// uint:
inline uint abs(uint x) { return x; }
inline uint abs_diff(uint x, uint y) { return x < y ? (y - x) : (x - y); }
uint add_sat(uint x, uint y);
inline uint hadd(uint x, uint y) { return (x >> 1) + (y >> 1) + (x & y & 1); }
uint rhadd(uint x, uint y);
inline uint clamp(uint x, uint minval, uint maxval) { return min(max(x, minval), maxval); }
uint clamp(uint x, int minval, int maxval);
uint clz(uint x);
uint ctz(uint x);
uint mad_hi(uint a, uint b, uint c);
uint mad_sat(uint a, uint b, uint c);
uint mul_hi(uint x, uint y);
uint rotate(uint v, uint i);
uint sub_sat(uint x, uint y);
inline uint popcount(uint x) { return __popc(x); }
ulong upsample(uint hi, uint lo);

// long:
inline ulong abs(long x) { return x >= 0 ? x : -x; }
inline ulong abs_diff(long x, long y) { return x < y ? (y - x) : (x - y); }
long add_sat(long x, long y);
inline long hadd(long x, long y) { return (x >> 1) + (y >> 1) + (x & y & 1); }
long rhadd(long x, long y);
inline long clamp(long x, long minval, long maxval) { return min(max(x, minval), maxval); }
long clz(long x);
long ctz(long x);
long mad_hi(long a, long b, long c);
long mad_sat(long a, long b, long c);
long mul_hi(long x, long y);
long rotate(long v, long i);
long sub_sat(long x, long y);
inline long popcount(long x) { return __popcll(reinterpret_cast<ulong&>(x)); }
long upsample(long hi, ulong lo);

// ulong:
inline ulong abs(ulong x) { return x; }
inline ulong abs_diff(ulong x, ulong y) { return x < y ? (y - x) : (x - y); }
ulong add_sat(ulong x, ulong y);
inline ulong hadd(ulong x, ulong y) { return (x >> 1) + (y >> 1) + (x & y & 1); }
ulong rhadd(ulong x, ulong y);
inline ulong clamp(ulong x, ulong minval, ulong maxval) { return min(max(x, minval), maxval); }
ulong clamp(ulong x, long minval, long maxval);
ulong clz(ulong x);
ulong ctz(ulong x);
ulong mad_hi(ulong a, ulong b, ulong c);
ulong mad_sat(ulong a, ulong b, ulong c);
ulong mul_hi(ulong x, ulong y);
ulong rotate(ulong v, ulong i);
ulong sub_sat(ulong x, ulong y);
inline ulong popcount(ulong x) { return __popcll(x); }

// TODO: Implement vectorized versions of the integer math functions above.
// For now we've only done upsample for n=2 and n=4

// The following functions need to be implemented for int's and their vectorizations: mad24, mul24

inline int mul24(int x, int y)        { return __mul24(x, y); }
inline int mad24(int x, int y, int z) { return __mul24(x, y) + z; }

// The following are available, for some reason, only for a specific combination of types:

// §6.15.3.1. Extended Bit Operations
// --------------------------------------------------------------

// The following should be implemented for the following types, and their vectorizations:
// char, uchar, short, ushort, int, uint, long, ulong.
//
// gentype bitfield_insert(gentype base, gentype insert, uint offset, uint count);
// igentype bitfield_extract_signed(gentype base, uint offset, uint count);
// ugentype bitfield_extract_unsigned(gentype base, uint offset, uint count);
// gentype bit_reverse(gentype base);

// char:
char bitfield_insert(char base, char insert, uint offset, uint count);
char bitfield_extract_signed(char base, uint offset, uint count);
uchar bitfield_extract_unsigned(char base, uint offset, uint count);
char bit_reverse(char base);

// uchar:
uchar bitfield_insert(uchar base, uchar insert, uint offset, uint count);
char bitfield_extract_signed(char base, uint offset, uint count);
uchar bitfield_extract_unsigned(uchar base, uint offset, uint count);
uchar bit_reverse(uchar base);

// short:
short bitfield_insert(short base, short insert, uint offset, uint count);
short bitfield_extract_signed(short base, uint offset, uint count);
ushort bitfield_extract_unsigned(short base, uint offset, uint count);
short bit_reverse(short base);

// ushort:
ushort bitfield_insert(ushort base, ushort insert, uint offset, uint count);
short bitfield_extract_signed(short base, uint offset, uint count);
ushort bitfield_extract_unsigned(ushort base, uint offset, uint count);
ushort bit_reverse(ushort base);

// int:
int bitfield_insert(int base, int insert, uint offset, uint count);
int bitfield_extract_signed(int base, uint offset, uint count);
uint bitfield_extract_unsigned(int base, uint offset, uint count);
int bit_reverse(int base);

// uint:
uint bitfield_insert(uint base, uint insert, uint offset, uint count);
int bitfield_extract_signed(int base, uint offset, uint count);
uint bitfield_extract_unsigned(uint base, uint offset, uint count);
uint bit_reverse(uint base);

// long:
long bitfield_insert(long base, long insert, uint offset, uint count);
long bitfield_extract_signed(long base, uint offset, uint count);
ulong bitfield_extract_unsigned(long base, uint offset, uint count);
long bit_reverse(long base);

// ulong:
ulong bitfield_insert(ulong base, ulong insert, uint offset, uint count);
long bitfield_extract_signed(long base, uint offset, uint count);
ulong bitfield_extract_unsigned(ulong base, uint offset, uint count);
ulong bit_reverse(ulong base);

// §6.15.4. Common Functions
// --------------------------------------------------------------

// The following should be implemented for the floating-point types
// half, float, double , and their vectorizations.
//
// gentype clamp(gentype x, gentype minval, gentype maxval) { return min(max(x, minval), maxval); }
// gentype degrees(gentype radians);
// gentype mix(gentype x, gentype y, gentype a) { return x + (y - x) * a; }
// gentype radians(gentype degrees);
// gentype step(gentype edge, gentype x);
// gentype smoothstep(gentype edge0, gentype edge1, gentype x);
// gentype sign(gentype x);
//
// No need to implement min and max - CUDA makes those available with the same name

// float:
inline float clamp(float x, float minval, float maxval) { return min(max(x, minval), maxval); }
float degrees(float radians);
inline float mix(float x, float y, float a) { return x + (y - x) * a; }
float radians(float degrees);
float step(float edge, float x);
float smoothstep(float edge0, float edge1, float x);
float sign(float x);

// double:
inline double clamp(double x, double minval, double maxval) { return min(max(x, minval), maxval); }
double degrees(double radians);
inline double mix(double x, double y, double a) { return x + (y - x) * a; }
double radians(double degrees);
double step(double edge, double x);
double smoothstep(double edge0, double edge1, double x);
double sign(double x);

// half:
inline half clamp(half x, half minval, half maxval) { return min(max(x, minval), maxval); }
half degrees(half radians);
inline half mix(half x, half y, half a) { return x + (y - x) * a; }
half radians(half degrees);
half step(half edge, half x);
half smoothstep(half edge0, half edge1, half x);
half sign(half x);

// §6.15.5. Geometric Functions
// --------------------------------------------------------------

// §6.15.6. Relational Functions
// --------------------------------------------------------------

// §6.15.7. Vector Data Load and Store Functions
// --------------------------------------------------------------

// §6.15.8. Synchronization Functions
// --------------------------------------------------------------

// void barrier(cl_mem_fence_flags flags);
// void work_group_barrier(cl_mem_fence_flags flags);
// void work_group_barrier(cl_mem_fence_flags flags, memory_scope scope);

// Note: Subgroups are not supported in CUDA

// §6.15.9. Legacy Explicit Memory Fence Functions
// --------------------------------------------------------------

// void mem_fence(cl_mem_fence_flags flags);
// void read_mem_fence(cl_mem_fence_flags flags);
// void write_mem_fence(cl_mem_fence_flags flags);

// §6.15.10. Address Space Qualifier Functions - not implemented,
// since CUDA does not use explicit address space qualifiers on
// pointers.

// §6.15.11. Async Copies From Global to Local Memory, Local to Global Memory, and Prefetch
// --------------------------------------------------------------

// §6.15.12. Atomic Functions
// --------------------------------------------------------------

// §6.15.13. Miscellaneous Vector Functions
// --------------------------------------------------------------

// §6.15.14. printf
// --------------------------------------------------------------

// §6.15.15. Image Read and Write Functions
// --------------------------------------------------------------

// §6.15.16. Work-group Collective Functions
// --------------------------------------------------------------

// §6.15.17. Work-group Collective Uniform Arithmetic Functions
// --------------------------------------------------------------

// §6.15.18. Pipe Functions
// --------------------------------------------------------------

// §6.15.19. Enqueuing Kernels
// --------------------------------------------------------------

// §6.15.20. Sub-Group Functions
// --------------------------------------------------------------

// §6.15.21. Kernel Clock Functions
// --------------------------------------------------------------


__device__ inline void barrier(int kind)
{
//    assert(kind == CLK_LOCAL_MEM_FENCE) noexcept;
    __syncthreads();
}


#endif // __OPENCL_VERSION__
#endif // PORT_FROM_OPENCL_BUILTIN_FUNCTIONS_CUH_
