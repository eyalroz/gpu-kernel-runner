/**
 * @file
 *
 * @brief Definitions and aliases of functions which are built-in for OpenCL,
 * but not for CUDA. These are mostly functions from §6.15 of the OpenCL
 * standard, as well as conversion 'operators' from §6.4.3 and §6.4.4.2 .
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
#include "opencl_vector_types.cuh"
#include "opencl_defines.cuh"

#include <vector_types.h>
#include <cuda_fp16.h>

// §6.4.3 Explicit conversions
// ===========================
// destType convert_destType<_sat><_roundingMode>(sourceType)
// destTypen convert_destTypen<_sat><_roundingMode>(sourceTypen)

// §6.4.4.2. Reinterpreting Types Using as_type() and as_typen()
// =============================================================

// "All data types described in Built-in Scalar Data Types and
// Built-in Vector Data Types (except bool, void, and half [19])
// may be also reinterpreted as another data type of the same size"


// §6.15.1. Work-Item Functions
// ============================

namespace detail_ {

constexpr __host__ __device__ inline unsigned int
get_dim3_element(const dim3& d3, int index) noexcept
{
    switch (index) {
        case 0:  return d3.x;
        case 1:  return d3.y;
        case 2:
        default: return d3.z;
    }
}

} // namespace detail_


/// Returns the number of dimensions in use. This is the value given to the work_dim argument specified in clEnqueueNDRangeKernel.
inline uint get_work_dim() noexcept { return 3; }

/// Returns the unique global work-item ID value for dimension identified by dimindx. The global work-item ID specifies the work-item ID based on the number of global work-items specified to execute the kernel.
/// Valid values of dimindx are 0 to get_work_dim() - 1. For other values of dimindx, get_global_id() returns 0.
inline size_t get_global_id(int dimension_index) noexcept
{
    // Note: We could have used:
    //
    //  return detail_::get_dim3_element(threadIdx, dimension_index) +
    //  detail_::get_dim3_element(blockIdx, dimension_index) *
    //  detail_::get_dim3_element(blockDim, dimension_index) noexcept;
    //
    // But I'm not sure we can trust the compiler to optimize
    // all of that away

    switch (dimension_index) {
        case 0:  return threadIdx.x + static_cast<size_t>(blockIdx.x) * blockDim.x;
        case 1:  return threadIdx.y + static_cast<size_t>(blockIdx.y) * blockDim.y;
        case 2:
        default: return threadIdx.z + static_cast<size_t>(blockIdx.z) * blockDim.z;
    }
}

/// Returns the number of local work-items specified in dimension identified by dimindx. This value is at most the value given by the local_work_size argument to clEnqueueNDRangeKernel if local_work_size is not NULL; otherwise the OpenCL implementation chooses an appropriate local_work_size value which is returned by this function. If the kernel is executed with a non-uniform work-group size [41], calls to this built-in from some work-groups may return different values than calls to this built-in from other work-groups.
/// Valid values of dimindx are 0 to get_work_dim() - 1. For other values of dimindx, get_local_size() returns 1.
inline size_t get_local_size(unsigned dimension_index) noexcept
{
    return detail_::get_dim3_element(blockDim, dimension_index);
}

/// Returns the same value as that returned by get_local_size(dimindx) if the kernel is executed with a uniform work-group size.
/// If the kernel is executed with a non-uniform work-group size, returns the number of local work-items in each of the work-groups that make up the uniform region of the global range in the dimension identified by dimindx. If the local_work_size argument to clEnqueueNDRangeKernel is not NULL, this value will match the value specified in local_work_size[dimindx]. If local_work_size is NULL, this value will match the local size that the implementation determined would be most efficient at implementing the uniform region of the global range.
/// Valid values of dimindx are 0 to get_work_dim() - 1. For other values of dimindx, get_enqueued_local_size() returns 1.
inline size_t get_enqueued_local_size(uint dimindx) noexcept;

/// Returns the unique local work-item ID, i.e. a work-item within a specific work-group for dimension identified by dimindx.
/// Valid values of dimindx are 0 to get_work_dim() - 1. For other values of dimindx, get_local_id() returns 0.
inline size_t get_local_id(int dimension_index) noexcept
{
    return detail_::get_dim3_element(threadIdx, dimension_index);
}

/// Returns the number of work-groups that will execute a kernel for dimension identified by dimindx.
/// Valid values of dimindx are 0 to get_work_dim() - 1. For other values of dimindx, get_num_groups() returns 1.
inline size_t get_num_groups(uint dimension_index) noexcept
{
    return detail_::get_dim3_element(gridDim, dimension_index);
}

/// Returns the number of global work-items specified for dimension identified by dimindx. This value is given by the global_work_size argument to clEnqueueNDRangeKernel.
/// Valid values of dimindx are 0 to get_work_dim() - 1. For other values of dimindx, get_global_size() returns 1.
inline size_t get_global_size(uint dimension_index) noexcept
{
    return static_cast<size_t>(get_num_groups(dimension_index)) * get_local_size(dimension_index);
}

/// get_group_id returns the work-group ID which is a number from 0 .. get_num_groups(dimindx) - 1.
/// Valid values of dimindx are 0 to get_work_dim() - 1. For other values, get_group_id() returns 0.
inline size_t get_group_id(int dimension_index) noexcept
{
    return detail_::get_dim3_element(blockIdx, dimension_index);
}

/**
 * get_global_offset returns the offset values specified in global_work_offset argument to clEnqueueNDRangeKernel.
 * Valid values of dimindx are 0 to get_work_dim() - 1. For other values, get_global_offset() returns 0.
 * Requires support for OpenCL C 1.1 or newer.
 * @param dimindx
 * @return
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
inline double sincos(double x, double *cosval) { double sinval; sincos(x, &sinval, cosval); return sinval; }
inline double atan2pi(double y, double x) noexcept { return atan2(y, x) * M_1_PI; }
inline double tanpi(double x) noexcept { return tan(M_PI * x); }

inline double2 remquo(double2 x, double2 y, int2 *quo) noexcept
{
    return { remquo(x.x, y.x, &quo->x), remquo(x.y, y.y, &quo->y) };
}

inline double4 remquo(double4 x, double4 y, int4 *quo) noexcept
{
    return {
        remquo(x.x, y.x, &quo->x),
        remquo(x.y, y.y, &quo->y),
        remquo(x.z, y.z, &quo->z),
        remquo(x.w, y.w, &quo->w)
    };
}

// TODO: Implement
// doublen remquo(doublen x, doublen y, intn *quo)
// for n=3,8,16

inline double fract(double x, double *iptr) noexcept
{
    double floor_ = floor(x);
    *iptr = floor_;
    return fmin(x - floor_, 0x1.fffffffffffffp-1);
}

inline double frexp(double x, int *exp) noexcept
{
    detail_::destructured_double x_;
    x_.value = x;
    *exp = x_.parts.exponent;
    return x_.parts.significand;
}

inline double2 frexp(double2 x, int2 *exp) noexcept
{
    return {
        frexp(x.x, &exp->x),
        frexp(x.y, &exp->y),
    };
}

inline double4 frexp(double4 x, int4 *exp) noexcept
{
    return {
        frexp(x.x, &exp->x),
        frexp(x.y, &exp->y),
        frexp(x.z, &exp->z),
        frexp(x.w, &exp->w),
    };
}

// TODO: implement frexp for doublen with n = 3,8,16

inline double2 ldexp(double2 x, int2 k) noexcept { return { ldexp(x.x, k.x), ldexp(x.y, k.y), }; }
inline double4 ldexp(double4 x, int4 k) noexcept { return { ldexp(x.x, k.x), ldexp(x.y, k.y), ldexp(x.z, k.z), ldexp(x.w, k.w) }; }

// TODO: implement ldexp for doublen with n = 3,8,16

// Not implemented: double2 nan(ulong2 nancode)
// double nan(ulong nancode);
double2 pow(double2 x, int2 y) { return { pow(x.x, y.x), pow(x.y, y.y) }; }
double4 pow(double4 x, int4 y) { return { pow(x.x, y.x), pow(x.y, y.y), pow(x.z, y.z), pow(x.w, y.w) }; }

inline double lgamma_r(double x, int *signp) noexcept
{
    detail_::destructured_float gamma_;
    gamma_.value = tgammaf(x);
    *signp = gamma_.parts.sign;
    gamma_.parts.sign = 0; // make it positive;
    return logf(gamma_.value);
}
double2 lgamma_r(double2 x, int2 *signp) { return { lgamma_r(x.x, &signp->x), lgamma_r(x.y, &signp->y) }; }
double4 lgamma_r(double4 x, int4 *signp) { return { lgamma_r(x.x, &signp->x), lgamma_r(x.y, &signp->y), lgamma_r(x.w, &signp->w), lgamma_r(x.w, &signp->w) }; }

inline int ilogb(double x) noexcept { return reinterpret_cast<detail_::destructured_double&>(x).parts.exponent; }
inline int2 ilogb(double2 x) noexcept { return { ilogb(x.x), ilogb(x.y) }; }
inline int4 ilogb(double4 x) noexcept { return { ilogb(x.x), ilogb(x.y), ilogb(x.z), ilogb(x.w) }; }

inline double maxmag(double x, double y) noexcept
{
    double abs_x = fabs(x);
    double abs_y = fabs(y);
    if (abs_x == abs_y) { return fmax(x,y); }
    return (abs_x > abs_y) ? x : y;
}
inline double minmag(double x, double y) noexcept
{
    double abs_x = fabs(x);
    double abs_y = fabs(y);
    if (abs_x == abs_y) { return fmin(x,y); }
    return (abs_x < abs_y) ? x : y;
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
    return fmin(x - floor_, 0x1.fffffep-1f);
}

inline float frexp(float x, int *exp) noexcept
{
    detail_::destructured_float x_;
    x_.value = x;
    *exp = x_.parts.exponent;
    return x_.parts.significand;
}

inline float2 frexp(float2 x, int2 *exp) noexcept
{
    return {
        frexp(x.x, &exp->x),
        frexp(x.y, &exp->y),
    };
}

inline float4 frexp(float4 x, int4 *exp) noexcept
{
    return {
        frexp(x.x, &exp->x),
        frexp(x.y, &exp->y),
        frexp(x.z, &exp->z),
        frexp(x.w, &exp->w),
    };
}
// TODO: implement frexp for floatn with n = 3,8,16
inline float hypot(float x, float y) noexcept { return hypotf(x, y); }
inline int ilogb(float x) noexcept { return reinterpret_cast<detail_::destructured_float&>(x).parts.exponent; }
inline int2 ilogb(float2 x) noexcept { return { ilogb(x.x), ilogb(x.y) }; }
inline int4 ilogb(float4 x) noexcept { return { ilogb(x.x), ilogb(x.y), ilogb(x.z), ilogb(x.w) }; }
inline float ldexp(float x, int k) noexcept { return ldexpf(x, k); }
inline float2 ldexp(float2 x, int k) noexcept { return { ldexpf(x.x, k), ldexpf(x.y, k) }; }
inline float4 ldexp(float4 x, int k) noexcept { return { ldexpf(x.x, k), ldexpf(x.y, k), ldexpf(x.z, k), ldexpf(x.w, k) }; }
inline float2 ldexp(float2 x, int2 k) noexcept { return { ldexpf(x.x, k.x), ldexpf(x.y, k.y) }; }
inline float4 ldexp(float4 x, int4 k) noexcept { return { ldexpf(x.x, k.x), ldexpf(x.y, k.y), ldexpf(x.z, k.z), ldexpf(x.w, k.w) }; }
// TODO: implement ldexp with vectorized and non-vectorized k for floatn with n = 3,8,16
inline float lgamma(float x) noexcept { return lgammaf(x); }
inline float lgamma_r(float x, int *signp) noexcept
{
    detail_::destructured_float gamma_;
    gamma_.value = tgammaf(x);
    *signp = gamma_.parts.sign;
    gamma_.parts.sign = 0; // make it positive;
    return logf(gamma_.value);
}
float2 lgamma_r(float2 x, int2 *signp) { return { lgamma_r(x.x, &signp->x), lgamma_r(x.y, &signp->y) }; }
float4 lgamma_r(float4 x, int4 *signp) { return { lgamma_r(x.x, &signp->x), lgamma_r(x.y, &signp->y), lgamma_r(x.w, &signp->w), lgamma_r(x.w, &signp->w) }; }
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
inline float2 pow(float2 x, int2 y) { return { pow(x.x, y.x), pow(x.y, y.y) }; }
inline float4 pow(float4 x, int4 y) { return { pow(x.x, y.x), pow(x.y, y.y), pow(x.z, y.z), pow(x.w, y.w) }; }
inline float powr(float x, float y) noexcept { return  exp2f(y * log2f(x)); }
inline float remainder(float x, float y) noexcept { return remainderf(x, y); }
inline float remquo(float x, float y, int *quo) noexcept { return remquof(x, y, quo); }
inline float2 remquo(float2 x, float2 y, int2 *quo) { return { remquo(x.x, y.x, &quo->x), remquo(x.y, y.y, &quo->y) }; }
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
    return fmin(x - floor_, 0x1.fffffep-1f);
}

inline half frexp(half x, int *exp) noexcept
{
    detail_::destructured_half x_;
    x_.value = x;
    *exp = x_.parts.exponent;
    return x_.parts.significand;
}

inline half2 frexp(half2 x, int2 *exp) noexcept
{
    return {
        frexp(x.x, &exp->x),
        frexp(x.y, &exp->y),
    };
}

#ifdef HAVE_HALF_4
inline half4 frexp(half4 x, int4 *exp) noexcept
{
return {
        frexp(x.x, &exp->x),
        frexp(x.y, &exp->y),
        frexp(x.z, &exp->z),
        frexp(x.w, &exp->w),
    };
}
#endif

// TODO: implement frexp for halfn with n = 3,8,16
inline half hypot(half x, half y) noexcept { return hypotf(x, y); }
inline int ilogb(half x) noexcept { return reinterpret_cast<detail_::destructured_half&>(x).parts.exponent; }
inline int2 ilogb(half2 x) noexcept { return { ilogb(x.x), ilogb(x.y) }; }
#ifdef HAVE_HALF_4
inline int4 ilogb(half4 x) noexcept { return { ilogb(x.x), ilogb(x.y), ilogb(x.z), ilogb(x.w) }; }
#endif
//inline half ldexp(half x, int k) noexcept { return hldexp(x, k); }
//inline half2 ldexp(half2 x, int k) noexcept { return { hldexp(x.x, k), hldexp(x.y, k) }; }
#ifdef HAVE_HALF_4
//inline half4 ldexp(half4 x, int k) noexcept { return { hldexp(x.x, k), hldexp(x.y, k), hldexp(x.z, k), hldexp(x.w, k) }; }
#endif
//inline half2 ldexp(half2 x, int2 k) noexcept { return { hldexp(x.x, k.x), hldexp(x.y, k.y) }; }
#ifdef HAVE_HALF_4
//inline half4 ldexp(half4 x, int4 k) noexcept { return { hldexp(x.x, k.x), hldexp(x.y, k.y), hldexp(x.z, k.z), hldexp(x.w, k.w) }; }
#endif
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
half2 lgamma_r(half2 x, int2 *signp) { return { lgamma_r(x.x, &signp->x), lgamma_r(x.y, &signp->y) }; }
#ifdef HAVE_HALF4
half4 lgamma_r(half4 x, int4 *signp) { return { lgamma_r(x.x, &signp->x), lgamma_r(x.y, &signp->y), lgamma_r(x.w, &signp->w), lgamma_r(x.w, &signp->w) }; }
#endif
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
inline half2 pow(half2 x, int2 y) { return { pow(x.x, y.x), pow(x.y, y.y) }; }
#ifdef HAVE_HALF_4
inline half4 pow(half4 x, int4 y) { return { pow(x.x, y.x), pow(x.y, y.y), pow(x.z, y.z), pow(x.w, y.w) }; }
#endif
//inline half powr(half x, half y) noexcept { return hpowr(x, y); }
// inline half remainder(half x, half y) noexcept { return hremainder(x, y); }
// inline half remquo(half x, half y, int *quo) noexcept { return hremquo(x, y, *quo); }
// inline half2 remquo(half2 x, half2 y, int2 *quo) { return { remquo(x.x, y.x, &quo->x), remquo(x.y, y.y, &quo->y) }; }
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
// abs, min, max .

// These functions should be implemented for each of:
//
// char, uchar, short, ushort, int, uint, long, ulong
// and for their vector types

// char:
uchar abs_diff(char x, char y) { return x < y ? (y - x) : (x - y); }
char add_sat(char x, char y);
char hadd(char x, char y) { return (x >> 1) + (y >> 1) + (x & y & 1); }
char rhadd(char x, char y);
char clamp(char x, char minval, char maxval) { return min(max(x, minval), maxval); }
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
uchar abs_diff(uchar x, uchar y) { return x < y ? (y - x) : (x - y); }
uchar add_sat(uchar x, uchar y);
uchar hadd(uchar x, uchar y) { return (x >> 1) + (y >> 1) + (x & y & 1); }
uchar rhadd(uchar x, uchar y);
uchar clamp(uchar x, uchar minval, uchar maxval) { return min(max(x, minval), maxval); }
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
ushort abs_diff(short x, short y) { return x < y ? (y - x) : (x - y); }
short add_sat(short x, short y);
short hadd(short x, short y) { return (x >> 1) + (y >> 1) + (x & y & 1); }
short rhadd(short x, short y);
short clamp(short x, short minval, short maxval) { return min(max(x, minval), maxval); }
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
ushort abs_diff(ushort x, ushort y) { return x < y ? (y - x) : (x - y); }
ushort add_sat(ushort x, ushort y);
ushort hadd(ushort x, ushort y) { return (x >> 1) + (y >> 1) + (x & y & 1); }
ushort rhadd(ushort x, ushort y);
ushort clamp(ushort x, ushort minval, ushort maxval) { return min(max(x, minval), maxval); }
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
uint abs_diff(int x, int y) { return x < y ? (y - x) : (x - y); }
int add_sat(int x, int y);
int hadd(int x, int y) { return (x >> 1) + (y >> 1) + (x & y & 1); }
int rhadd(int x, int y);
int clamp(int x, int minval, int maxval) { return min(max(x, minval), maxval); }
int clz(int x);
int ctz(int x);
int mad_hi(int a, int b, int c);
int mad_sat(int a, int b, int c);
int mul_hi(int x, int y);
int rotate(int v, int i);
int sub_sat(int x, int y);
int popcount(int x) { return __popc(reinterpret_cast<uint&>(x)); }
long upsample(int hi, uint lo);

// uint:
uint abs_diff(uint x, uint y) { return x < y ? (y - x) : (x - y); }
uint add_sat(uint x, uint y);
uint hadd(uint x, uint y) { return (x >> 1) + (y >> 1) + (x & y & 1); }
uint rhadd(uint x, uint y);
uint clamp(uint x, uint minval, uint maxval) { return min(max(x, minval), maxval); }
uint clamp(uint x, int minval, int maxval);
uint clz(uint x)
uint ctz(uint x);
uint mad_hi(uint a, uint b, uint c);
uint mad_sat(uint a, uint b, uint c);
uint mul_hi(uint x, uint y);
uint rotate(uint v, uint i);
uint sub_sat(uint x, uint y);
uint popcount(uint x) { return __popc(x); }
ulong upsample(uint hi, uint lo);

// long:
ulong abs_diff(long x, long y) { return x < y ? (y - x) : (x - y); }
long add_sat(long x, long y);
long hadd(long x, long y) { return (x >> 1) + (y >> 1) + (x & y & 1); }
long rhadd(long x, long y);
long clamp(long x, long minval, long maxval) { return min(max(x, minval), maxval); }
long clz(long x);
long ctz(long x);
long mad_hi(long a, long b, long c);
long mad_sat(long a, long b, long c);
long mul_hi(long x, long y);
long rotate(long v, long i);
long sub_sat(long x, long y);
long popcount(long x) { return __popcll(reinterpret_cast<ulong&>(x)); }
long upsample(long hi, ulong lo);

// ulong:
ulong abs_diff(ulong x, ulong y) { return x < y ? (y - x) : (x - y); }
ulong add_sat(ulong x, ulong y);
ulong hadd(ulong x, ulong y) { return (x >> 1) + (y >> 1) + (x & y & 1); }
ulong rhadd(ulong x, ulong y);
ulong clamp(ulong x, ulong minval, ulong maxval) { return min(max(x, minval), maxval); }
ulong clamp(ulong x, long minval, long maxval);
ulong clz(ulong x);
ulong ctz(ulong x);
ulong mad_hi(ulong a, ulong b, ulong c);
ulong mad_sat(ulong a, ulong b, ulong c);
ulong mul_hi(ulong x, ulong y);
ulong rotate(ulong v, ulong i);
ulong sub_sat(ulong x, ulong y);
ulong popcount(ulong x) { return __popcll(x); }

// TODO: Implement vectorized versions of the integer math functions above.
// For now we've only done upsample for n=2 and n=4

short2  upsample(char2   hi, uchar2 lo) { return { upsample((char) hi.x, lo.x), upsample((char) hi.y, lo.y) }; }
ushort2 upsample(uchar2  hi, uchar2 lo) { return { upsample(hi.x, lo.x), upsample(hi.y, lo.y) }; }
int2    upsample(short2  hi, short2 lo) { return { upsample(hi.x, lo.x), upsample(hi.y, lo.y) }; }
uint2   upsample(ushort2 hi, short2 lo) { return { upsample(hi.x, lo.x), upsample(hi.y, lo.y) }; }
long2   upsample(int2    hi, int2   lo) { return { upsample(hi.x, lo.x), upsample(hi.y, lo.y) }; }
ulong2  upsample(uint2   hi, int2   lo) { return { upsample(hi.x, lo.x), upsample(hi.y, lo.y) }; }

short4 upsample(char4 hi, uchar4 lo) {
    return {
        upsample((char) hi.x, lo.x), upsample((char) hi.y, lo.y),
        upsample((char) hi.z, lo.z), upsample((char) hi.w, lo.w)
    };
}

ushort4 upsample(uchar4 hi, uchar4 lo) {
    return {
        upsample(hi.x, lo.x), upsample(hi.y, lo.y),
        upsample(hi.z, lo.z), upsample(hi.w, lo.w)
    };
}
int4 upsample(short4 hi, short4 lo) {
    return {
        upsample(hi.x, lo.x), upsample(hi.y, lo.y),
        upsample(hi.z, lo.z), upsample(hi.w, lo.w)
    };
}
uint4 upsample(ushort4 hi, short4 lo) {
    return {
        upsample(hi.x, lo.x), upsample(hi.y, lo.y),
        upsample(hi.z, lo.z), upsample(hi.w, lo.w)
    };
}
long4 upsample(int4 hi, int4 lo) {
    return {
        upsample(hi.x, lo.x), upsample(hi.y, lo.y),
        upsample(hi.z, lo.z), upsample(hi.w, lo.w)
    };
}
ulong4 upsample(uint4 hi, int4 lo) {
    return {
        upsample(hi.x, lo.x), upsample(hi.y, lo.y),
        upsample(hi.z, lo.z), upsample(hi.w, lo.w)
    };
}

// The following functions need to be implemented for int's and their vectorizations: mad24, mul24

int mul24(int x, int y)        { return __mul24(x, y); }
int mad24(int x, int y, int z) { return __mul24(x, y) + z; }
// TODO: Implement vectorizations of mul24, mad24

// The following are available, for some reason, only for a specific combination of types:

// TODO: Implement these
uint dot(uchar4 a, uchar4 b);
int dot(char4 a, char4 b);
int dot(uchar4 a, char4 b);
int dot(char4 a, uchar4 b);
uint dot_acc_sat(uchar4 a, uchar4 b, uint acc);
int dot_acc_sat(char4 a, char4 b, int acc);
int dot_acc_sat(uchar4 a, char4 b, int acc);
int dot_acc_sat(char4 a, uchar4 b, int acc);
uint dot_4x8packed_uu_uint(uint a, uint b);
int dot_4x8packed_ss_int(uint a, uint b);
int dot_4x8packed_us_int(uint a, uint b);
int dot_4x8packed_su_int(uint a, uint b);
uint dot_acc_sat_4x8packed_uu_uint(uint a, uint b, uint acc);
int dot_acc_sat_4x8packed_ss_int(uint a, uint b, int acc);
int dot_acc_sat_4x8packed_us_int(uint a, uint b, int acc);
int dot_acc_sat_4x8packed_su_int(uint a, uint b, int acc);

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
float clamp(float x, float minval, float maxval) { return min(max(x, minval), maxval); }
float degrees(float radians);
float mix(float x, float y, float a) { return x + (y - x) * a; }
float radians(float degrees);
float step(float edge, float x);
float smoothstep(float edge0, float edge1, float x);
float sign(float x);

// double:
double clamp(double x, double minval, double maxval) { return min(max(x, minval), maxval); }
double degrees(double radians);
double mix(double x, double y, double a) { return x + (y - x) * a; }
double radians(double degrees);
double step(double edge, double x);
double smoothstep(double edge0, double edge1, double x);
double sign(double x);

// half:
half clamp(half x, half minval, half maxval) { return min(max(x, minval), maxval); }
half degrees(half radians);
half mix(half x, half y, half a) { return x + (y - x) * a; }
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
