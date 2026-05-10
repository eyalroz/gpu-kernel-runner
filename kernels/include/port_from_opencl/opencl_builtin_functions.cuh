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
 * @todo
 * 1. Double-check all definitions in this file respect PORT_FROM_OPENCL_ENABLE_HALF_PRECISION
 * 2. Figure out where to get the definition of memory_scope from.
 * 3. Mark non-template functions as inline
 *
 */
#ifndef PORT_FROM_OPENCL_BUILTIN_FUNCTIONS_CUH_
#define PORT_FROM_OPENCL_BUILTIN_FUNCTIONS_CUH_

#ifndef __OPENCL_VERSION__

#include <math.h>

#include "opencl_scalar_types.cuh"
#include "opencl_defines.cuh"

#include <vector_types.h>

#include <type_traits>
#include <spdlog/fmt/bundled/format.h>

// §6.4.3 Explicit conversions
// ===========================

/**
 * The two forms of explicit conversion functions are:
 * For scalar types:
 *
 *   destType convert_destType<_sat><_roundingMode>(sourceType)
 *
 * For vectorized types with length n:
 *
 *   destTypen convert_destTypen<_sat><_roundingMode>(sourceTypen)
 *
 * ... and the basic types to/from we can convert are:
 * char, uchar, short, ushort, int, uint, long, ulong, float
 *
 * To avoid massive repetitions, we'll be using macros and templates to define
 * the different conversion functions
 */

#define PORT_FROM_OPENCL_DEFINE_FLOAT_TO_INT_CONVERSIONS(src, tgt) \
inline tgt convert_ ## tgt ## _rtz(src v) { return static_cast<tgt>(__ ## src ## 2 ## tgt ## _rz(v)); } \
inline tgt convert_ ## tgt ## _rte(src v) { return static_cast<tgt>(__ ## src ## 2 ## tgt ## _rn(v)); } \
inline tgt convert_ ## tgt ## _rtp(src v) { return static_cast<tgt>(__ ## src ## 2 ## tgt ## _ru(v)); } \
inline tgt convert_ ## tgt ## _rtn(src v) { return static_cast<tgt>(__ ## src ## 2 ## tgt ## _rd(v)); } \
inline template<> tgt convert_ ## tgt <src>(src v) { return convert_ ## tgt ## _rtz(v); } \

#ifdef PORT_FROM_OPENCL_ENABLE_HALF_PRECISION
#define PORT_FROM_OPENCL_DEFINE_HALF_TO_INT_CONVERSIONS(tgt) PORT_FROM_OPENCL_DEFINE_FLOAT_TO_INT_CONVERSIONS(half, tgt)
#else
#define PORT_FROM_OPENCL_DEFINE_HALF_TO_INT_CONVERSIONS(tgt)
#endif

#define PORT_FROM_OPENCL_DEFINE_CONVERSIONS_TO_INT_TYPE(tgt) \
template <typename Source> \
tgt convert_ ## tgt (Source v) noexcept { return static_cast<tgt>(v); } \
PORT_FROM_OPENCL_DEFINE_HALF_TO_INT_CONVERSIONS(half, tgt) \
PORT_FROM_OPENCL_DEFINE_FLOAT_TO_INT_CONVERSIONS(float, tgt) \
PORT_FROM_OPENCL_DEFINE_FLOAT_TO_INT_CONVERSIONS(double, tgt)

#ifdef PORT_FROM_OPENCL_ENABLE_HALF_PRECISION
#define PORT_FROM_OPENCL_DEFINE_CONVERSIONS_TO_CHAR_TYPE(tgt) \
template <typename Source> \
tgt convert_ ## tgt (Source v) noexcept { return static_cast<tgt>(v); } \
inline tgt convert_ ## tgt ## _rtz(half v) { return static_cast<tgt>(__ ## half ## 2 ## tgt ## _rz(v)); } \
template<> tgt convert_ ## tgt <half>(half v) noexcept { return convert_ ## tgt ## _rtz(v); }
// Note: CUDA does not offer intrinsics for converting float or double to char-sized types
#else // PORT_FROM_OPENCL_ENABLE_HALF_PRECISION
#define PORT_FROM_OPENCL_DEFINE_CONVERSIONS_TO_CHAR_TYPE(tgt) \
template <typename Source> \
tgt convert_ ## tgt (Source v) noexcept { return static_cast<tgt>(v); }
#endif // PORT_FROM_OPENCL_ENABLE_HALF_PRECISION


// TODO: Define _sat conversions

PORT_FROM_OPENCL_DEFINE_CONVERSIONS_TO_CHAR_TYPE(char)
PORT_FROM_OPENCL_DEFINE_CONVERSIONS_TO_CHAR_TYPE(uchar)
PORT_FROM_OPENCL_DEFINE_CONVERSIONS_TO_INT_TYPE(short)
PORT_FROM_OPENCL_DEFINE_CONVERSIONS_TO_INT_TYPE(ushort)
PORT_FROM_OPENCL_DEFINE_CONVERSIONS_TO_INT_TYPE(int)
PORT_FROM_OPENCL_DEFINE_CONVERSIONS_TO_INT_TYPE(uint)
PORT_FROM_OPENCL_DEFINE_CONVERSIONS_TO_INT_TYPE(long)
PORT_FROM_OPENCL_DEFINE_CONVERSIONS_TO_INT_TYPE(ulong)

// do saturated versions


// §6.4.4.2. Reinterpreting Types Using as_type() and as_typen()
// =============================================================

// "All data types described in Built-in Scalar Data Types and
// Built-in Vector Data Types (except bool, void, and half [19])
// may be also reinterpreted as another data type of the same size"

#define PORT_FROM_OPENCL_DEFINE_ASTYPE(tgt) \
template <typename src> \
inline tgt as_ ## tgt (src v) \
{ \
    static_assert(sizeof(tgt) == sizeof(src), "as_type for types of different size is not supported"); \
    return reinterpret_cast<tgt>(v); \
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

// TODO: Allow for interpreting 4-component vector-types as 3-component OpenCL types, see §6.4.4.2

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

/**
 * Returns the unique global work-item ID value for one of the launch grid axes.
 *
 * @param[in] dimindx a value between 0 to get_work_dim() - 1. For an axis index outside
 * this range, get_global_id() returns 0.
 */
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

/**
 * Returns the number of local work-items in one of the kernel's launch axes
 *
* @param[in] dimindx a value between 0 to get_work_dim() - 1. For an axis index outside
 * this range, get_global_id() returns 1.
 *
 * @return The value of `local_work_size` specified for axis @p dimindx on launch -
 * except if local_work_size is lost; otherwise the OpenCL implementation chooses
 * an appropriate local_work_size value which is returned by this function.  calls to this
 * built-in from some work-groups may return different values than calls to this
 * built-in from other work-groups.
 */
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

// TODO: The clang parser used in some IDEs seems to include __clang_cuda_math.h and
//  __clang_cuda_runtime_wrapper.h ; and they're annoying!

// Table 11 functions

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

// Default, multi-type implementations of some generics (which are not
// offered by CUDA directly)

///@{
namespace detail_ {
/// A forward declaration; see section 6.15.6 below. Also, this
/// does not have the same declaration as the official isnan(),
/// and is defined for all numeric types, not just floating-point
/// ones
template <typename T> bool isnan(X x) noexcept;
} // namespace detail_
///@}

/**
 * Absolute value
 *
 * @note Floating-point types, which have values not comparable with 0,
 * as well as negative and positive 0, need a specialization of the general
 * implementation. CUDA offers fabs()-like functions for these types: double,
 * float, half
 *
 * @note For unsigned types, we rely on compiler optimization to remove the
 * condition check, which always holds in that case.
 *
 */
template <typename T> T fabs(T x) noexcept { return x >= 0 ? x : -x; }

/// Maximum magnitude between two values
template <typename T>
T fmax(T x, T y) noexcept
{
    if (::std::is_floating_point<T>::value and detail_::isnan(x)) { return y; }
    if (::std::is_floating_point<T>::value and detail_::isnan(y)) { return x; }
    return x < y ? y : x;
}

/// Maximum magnitude between two values
template <typename T>
T fmin(T x, T y) noexcept
{
    if (::std::is_floating_point<T>::value and detail_::isnan(x)) { return y; }
    if (::std::is_floating_point<T>::value and detail_::isnan(y)) { return x; }
    return y < x ? y : x;
}

/// Maximum absolute value amongst two values
template <typename T>
T maxmag(T x, T y) noexcept
{
    double abs_x = fabs(x);
    double abs_y = fabs(y);
    if (abs_x == abs_y) { return fmax(x,y); }
    return (abs_x > abs_y) ? x : y;
}

/// Minimum absolute value amongst two values
template <typename T>
T minmag(T x, T y) noexcept
{
    double abs_x = fabs(x);
    double abs_y = fabs(y);
    if (abs_x == abs_y) { return fmin(x,y); }
    return (abs_x < abs_y) ? x : y;
}


// The following double-precision functions are available directly in CUDA and need no adaptation:
//
// acos, acosh, acospi, asin, asinh, asinpi, atan, atan2, atanh, atanpi,
// atan2pi, cbrt, ceil, copysign, cos, cosh, cospi, erfc, erf, exp,
// exp2, exp10, expm1, fabs, fdim, floor, fma, fmax, fmin, fmod,
// ldexp, lgamma, nextafter, pow, remainder, remquo, rint, round,
// rsqrt, sin, sinh, sinpi, sqrt, tan, tanh, tgamma, trunc
//
// nan, sincos - provided, but with different signature

// double-precision parameter, but missing in CUDA
// ------------------------------------------------

inline double acospi(double x) noexcept { return acos(x) * M_1_PI; }
inline double asinpi(double x) noexcept { return asin(x) * M_1_PI; }
inline double atan2pi(double y, double x) noexcept { return atan2(y, x) * M_1_PI; }

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

inline double lgamma_r(double x, int *signp) noexcept
{
    detail_::destructured_float gamma_;
    gamma_.value = tgammaf(x);
    *signp = gamma_.parts.sign;
    gamma_.parts.sign = 0; // make it positive;
    return logf(gamma_.value);
}

inline int ilogb(double x) noexcept { return reinterpret_cast<detail_::destructured_double&>(x).parts.exponent; }

inline double mad(double a, double b, double c) noexcept { return fma(a, b, c); }
// TODO: implement this nan(), by taking the single bytes of the uint and constructing
// a char buffer to pass to the CUDA-style nan(char const *); see:
// https://stackoverflow.com/a/79906224/1593077
double nan(uint nancode) noexcept;

// Unfortunately, it doesn't seem CUDA offers something smarter for pown and powr than just pow...
inline double pown(double x, int y) { return pow(x,y); }
inline double powr(double x, double y) { return pow(x, y); }
inline double rootn(double x, int y) { return pow(x, 1.0/y); }

// Note the different signatures of the OpenCL-spec and CUDA-provided sincos()!
inline double sincos(double x, double *cosval) { double sinval; sincos(x, &sinval, cosval); return sinval; }
inline double sinpi(double x) noexcept { return sin(M_PI * x); }
inline double tanpi(double x) noexcept { return tan(M_PI * x); }


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
inline float cospi(float x) { return cospif(x); }
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
// Implemented generically:  gentyped fmax(gentyped x, float y)
inline float fmin(float x, float y) noexcept { return fminf(x, y); }
// Implemented generically: gentyped fmin(gentyped x, float y)
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
inline float modf(float x, float *iptr) noexcept { return modff(x, iptr); }
// TODO: implement this nan(), by taking the single bytes of the uint and constructing
// a char buffer to pass to the CUDA-style nan(char const *); see:
// https://stackoverflow.com/a/79906224/1593077
float nan(uint nancode) noexcept;
inline float nextafter(float x, float y) noexcept { return nextafterf(x, y); }
inline float pow(float x, float y) noexcept { return powf(x, y); }
inline float pown(float x, int y) noexcept { return  powf(x, y); }
inline float powr(float x, float y) noexcept { return  powf(x, y); }
inline float remainder(float x, float y) noexcept { return remainderf(x, y); }
inline float remquo(float x, float y, int *quo) noexcept { return remquof(x, y, quo); }
inline float rint(float x) noexcept { return rintf(x); }
inline double rootn(float x, int y) { return powf(x, 1.0/y); }
inline float round(float x) noexcept { return roundf(x); }
inline float rsqrt(float x) { return rsqrtf(x); }
inline float sin(float x) noexcept { return sinf(x); }
inline float sincos(float x, float *cosval) { float sinval; sincosf(x, &sinval, cosval); return sinval; }
inline float sinh(float x) noexcept { return sinhf(x); }
inline float sinpi(float x) noexcept { return sinf(M_PI_F * x); }
inline float sqrt(float x) noexcept { return sqrtf(x); }
inline float tan(float x) noexcept { return tanf(x); }
inline float tanh(float x) noexcept { return tanhf(x); }
inline float tanpi(float x) noexcept { return tanf(M_PI_F * x); }
inline float tgamma(float x) noexcept { return tgammaf(x); }
inline float trunc(float x) noexcept { return truncf(x); }

#ifdef PORT_FROM_OPENCL_ENABLE_HALF_PRECISION
// half-precision parameter
// -------------------------

// Note: CUDA (as of 13.2) does not provide half-precision versions of most of the
// math functions it offers for float- and double-precision. Also, we can't
// reasonably implement some of the missing functions using alternative functions,
// like we've done for the float and double types, since those implementations rely
// on other missing functions.
//
// So, the following are missing:
//
// acos, acosh, acospi, asin, asinh, asinpi, atan, atan2, atanh, atanpi, atan2pi
// cbrt, copysign, cosh, cospi, erf, erfc, expm1, fabs, fdim, fma, fmax, fmin, fmod,
// ldexp, lgamma, lgamma_r, log1p, logb, mad, modf, nextafter, pow, powr, pown,
// remainder, remquo, root, rootn, round, sincos, sinh, tan, tanh, tanpi, tgamma,
// sinpi, cospi


inline half ceil(half x) noexcept { return hceil(x); }
inline half cos(half x) noexcept { return hcos(x); }
inline half exp(half x) noexcept { return hexp(x); }
inline half exp2(half x) noexcept { return hexp2(x); }
inline half exp10(half x) noexcept { return hexp10(x); }
inline half floor(half x) noexcept { return hfloor(x); }
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

inline half hypot(half x, half y) noexcept { return hypotf(x, y); }
inline int ilogb(half x) noexcept { return reinterpret_cast<detail_::destructured_half&>(x).parts.exponent; }
inline half log(half x) noexcept { return hlog(x); }
inline half log2(half x) noexcept { return hlog2(x); }
inline half log10(half x) noexcept { return hlog10(x); }
inline half rint(half x) noexcept { return hrint(x); }
inline half sin(half x) noexcept { return hsin(x); }
inline half sqrt(half x) noexcept { return hsqrt(x); }
inline half trunc(half x) noexcept { return htrunc(x); }
inline half rsqrt(half x) { return hrsqrt(x); }


// allowed-half-precision functions (10-bit accuracy at least) from Table 12
// ------------------------------------------------------------------------

// CUDA doesn't expose lower-precision operations, so let's be naive here

template <typename T> T half_cos(T x) { return cos(x); }
template <typename T> T half_divide(T x, T y) { return x/y; }
template <typename T> T half_exp(T x) { return exp(x); }
template <typename T> T half_exp2(T x) { return exp2(x); }
template <typename T> T half_exp10(T x) { return exp10(x); }
template <typename T> T half_log(T x) { return log(x); }
template <typename T> T half_log2(T x) { return log2(x); }
template <typename T> T half_log10(T x) { return log10(x); }
template <typename T> T half_powr(T x, T y) { return powr(x); }
template <typename T> T half_recip(T x) { return 1/x; }
template <typename T> T half_rsqrt(T x) { return rsqrt(x); }
template <typename T> T half_sin(T x) { return sin(x); }
template <typename T> T half_sqrt(T x) { return sqrt(x); }
template <typename T> T half_tan(T x) { return tan(x); }


// native-device-instruction functions (10-bit accuracy at least) from Table 12
// ----------------------------------------------------------------------------

// We could specialize some of these functions for some of these types, perhaps,
// but... let's be naive for now.

template <typename T> T native_cos(T x) { return cos(x); }
template <typename T> T native_divide(T x, T y) { return x/y; }
template <typename T> T native_exp(T x) { return exp(x); }
template <typename T> T native_exp2(T x) { return exp2(x); }
template <typename T> T native_exp10(T x) { return exp10(x); }
template <typename T> T native_log(T x) { return log(x); }
template <typename T> T native_log2(T x) { return log2(x); }
template <typename T> T native_log10(T x) { return log10(x); }
template <typename T> T native_powr(T x, T y) { return powr(x); }
template <typename T> T native_recip(T x) { return 1/x; }
template <typename T> T native_rsqrt(T x) { return rsqrt(x); }
template <typename T> T native_sin(T x) { return sin(x); }
template <typename T> T native_sqrt(T x) { return sqrt(x); }
template <typename T> T native_tan(T x) { return tan(x); }

#endif // PORT_FROM_OPENCL_ENABLE_HALF_PRECISION

// §6.15.3. Integer Functions
// --------------------------------------------------------------

// Some of the OpenCL integer-math functions have CUDA equivalents with the exact same
// names - either with an identical signature or through implicit conversions. These are:
// abs, min, max .

// These functions should be implemented for each of:
//
// char, uchar, short, ushort, int, uint, long, ulong
// and for their vector types

namespace detail_ {

template <typename T>
size_t num_bits() { return sizeof(T) * CHAR_BIT; }

template <typename I>
using make_unsigned_t = typename ::std::make_unsigned<I>::type;

template <typename I>
using enable_if_t = typename ::std::enable_if<I>::type;

template <typename I>
using enable_if_integral_t = enable_if_t<::std::is_integral<I>>;

template <typename I> I ffs_nonzero(I x)
{
    static_assert(sizeof(I) == 1 or sizeof(I) == 2 sizeof(I) == 4 or sizeof(I) == 8, "Unsupported type size for clz");

    return (sizeof(I) <= sizeof(int)) ? __ffs(x) : __ffsll(x);
}

template <typename I> I ffs(I x)
{
    static_assert(sizeof(I) == 1 or sizeof(I) == 2 sizeof(I) == 4 or sizeof(I) == 8, "Unsupported type size for clz");

    I ffs_val = sizeof(I) <= sizeof(int) ? __ffs(x) : __ffsll(x);
    if (sizeof(I) == 4 or sizeof(I) == 8) { return ffs_val; }
    if (ffs_val < num_bits<I>()) { return ffs_val ; }
    return num_bits<I>();
}

} // namespace detail_

// , typename = detail_::enable_if_integral_t<I>
template <typename I> detail_::make_unsigned_t<I> abs(I x) { return x >= 0 ? x : -x; }
template <typename I> detail_::make_unsigned_t<I> abs_diff(I x) { return (x >= y) ? (x - y) : (y - x); }
// char add_sat(char x, char y);
template <typename I> I hadd(I x, I y) { return (x >> 1) + (y >> 1) + (x & y & 1); }
template <typename I> I rhadd(I x, I y) { return (x >> 1) + (y >> 1) + (x & 1 + y & 1 + 1) >> 1; }
template <typename I> I clamp(I x, I minval, I maxval) { return min(max(x, minval), maxval); }
template <typename I> I clz(I x)
{
    static_assert(::std::is_integral<I>::value);
    static_assert(sizeof(I) == 1 or sizeof(I) == 2 sizeof(I) == 4 or sizeof(I) == 8, "Unsupported type size for clz");
    switch (sizeof(I)) {
        case 1: return __clz(reinterpret_cast<uchar >(x)) - 24; // note the implicit promotion to int, but without sign-extension
        case 2: return __clz(reinterpret_cast<ushort>(x)) - 16; // note the implicit promotion to int, but without sign-extension
        case 4: return __clz(x); // unsigned int interpreted as int
        default:
        case 8: return __clzll(x);
    }
}
template <typename I> I ctz(I x) { return (x == 0) ? detail_::num_bits<I>() : detail_::ffs_nonzero(x); }
template <typename I> I mul_hi(I a, I b)
{
    // TODO: Consider using mulhi and umulhi, __mul64hi and __umul64hi !
    static_assert(::std::is_unsigned<I>::value, "Only unsigned types supported for now");
    static constexpr auto half_bits = detail_::num_bits<I>() / 2;
    if (::std::is_unsigned<I>::value) {
        return
            (((a >> half_bits) * b + a * (b >> half_bits)) >> half_bits)
            - (a >> half_bits) * (b >> half_bits);
    }
    // if (::std::is_signed<I>::value) {
    //     bool sign_a = a < 0;
    //     bool sign_b = b < 0;
    //     if (sign_a) { a = -a; }
    //     if (sign_b) { b = -b; }
    //     auto output_sign = sign_a ^ sign_b; // no logical xor, alas
    //     auto output_abs =
    //         (((a >> half_bits) * b + a * (b >> half_bits)) >> half_bits)
    //         - (a >> half_bits) * (b >> half_bits);
    //     return output_sign ? -output_abs : output_abs;
    // }
}
template <typename I> I mad_hi(I a, I b, I c)
{
    // TODO: Make sure everything has modulo-2^n semantics...
    struct { I hi, lo; } parts = { a * b, mul_hi(a,b) };
    if (c > ::std::numeric_limits<I>::max() - parts.lo) { return parts.hi + 1; }
    if (::std::is_signed<I>::value) {
        if (c < 0 and parts.lo - ::std::numeric_limits<I>::in() > -c) { return parts.hi - 1; }
    }
    return parts.hi;
}
template <typename I> I mad_sat(I a, I b, I c)
{
    static_assert(::std::is_unsigned<I>::value, "Only unsigned types supported for now");
    if (mul_hi(a, b) > 0) { return ::std::numeric_limits<I>::max(); }
    I m = a * b;
    I maybe_result = m + c;
    if ((m + c) < m) { return ::std::numeric_limits<I>::max(); }
    return m + c;
}
template <typename I> I rotate(I v, I i)
{
    auto num_bits = detail_::num_bits<I>();
    auto i = i % num_bits;
    return (v << i) & (v >> (num_bits - i)); // What will this do for signed types?
}
template <typename I> I sub_sat(I x, I y)
{
    if (::std::is_unsigned<I>::value) {
        return (y > x ? 0 : x - y);
    }
    if (y > 0) {
        return (y > x -::std::numeric_limits<I>::min()) ? ::std::numeric_limits<I>::min() : x - y;
    }
    return ((x - y) < x) ? ::std::numeric_limits<I>::max() : x - y;
}
template <typename I> I upsample(I x);
template <typename I> I popcount(I x)
{
    using UI = typename ::std::make_unsigned<I>::type;
    UI x_ = x;
    return (sizeof(I) <= sizeof(int)) ? __popc(x_) : __popcll(x_);
}

// specializations...

short  upsample(char   hi, uchar  lo) { return ((short ) hi) * (short ) (1 << (sizeof(char  ) * CHAR_BIT)) + lo; }
ushort upsample(uchar  hi, uchar  lo) { return ((ushort) hi) * (ushort) (1 << (sizeof(uchar ) * CHAR_BIT)) + lo; }
int    upsample(short  hi, ushort lo) { return ((int   ) hi) * (int   ) (1 << (sizeof(short ) * CHAR_BIT)) + lo; }
uint   upsample(ushort hi, ushort lo) { return ((uint  ) hi) * (uint  ) (1 << (sizeof(ushort) * CHAR_BIT)) + lo; }
long   upsample(int    hi, uint   lo) { return ((long  ) hi) * (long  ) (1 << (sizeof(int   ) * CHAR_BIT)) + lo; }
ulong  upsample(uint   hi, uint   lo) { return ((ulong ) hi) * (ulong ) (1 << (sizeof(uint  ) * CHAR_BIT)) + lo; }
template <typename I> I upsample(I x) = delete;

// TODO: Implement vectorized versions of the integer math functions above.
// For now we've only done upsample for n=2 and n=4

// Built-in 24-bit Integer Functions (Table 14)

// The following functions need to be implemented for int's

int mul24(int x, int y)        { return __mul24(x, y); }
int mad24(int x, int y, int z) { return __mul24(x, y) + z; }
// TODO: Implement vectorizations of mul24, mad24

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

#ifdef PORT_FROM_OPENCL_ENABLE_HALF_PRECISION
// half:
half clamp(half x, half minval, half maxval) { return min(max(x, minval), maxval); }
half degrees(half radians);
half mix(half x, half y, half a) { return x + (y - x) * a; }
half radians(half degrees);
half step(half edge, half x);
half smoothstep(half edge0, half edge1, half x);
half sign(half x);
#endif // PORT_FROM_OPENCL_ENABLE_HALF_PRECISION

// §6.15.5. Geometric Functions
// --------------------------------------------------------------

// §6.15.6. Relational Functions
// --------------------------------------------------------------

// generic / multi-type

template <typename T> int isequal(T x, T y) noexcept { return x == y; }
template <typename T> int isnotequal(T x, T y) noexcept { return x != y; }
template <typename T> int isgreater(T x, T y) noexcept { return x > y; }
template <typename T> int isgreaterequal(T x, T y) noexcept { return x >= y; }
template <typename T> int islesser(T x, T y) noexcept { return x < y; }
template <typename T> int islessequal(T x, T y) noexcept { return x <= y; }
template <typename T> int islessgreater(T x, T y) noexcept { return x < y or x > y; }

namespace detail_ {
template <typename I> I msb_mask() noexcept { return 1 << (num_bits<I>() - 1); }
template <typename I> I msb_of(I x) noexcept { return x & msb_mask<I>() ? 1 : 0; }
}

template <typename I> int any(I x) { return detail_::msb_of(x); } // this is meaningful mostly for vector types
template <typename I> int all(I x) { return detail_::msb_of(x); } // this is meaningful mostly for vector types
template <typename I> I bitselect(I a, I b, I c) { return (a & c) | (b & ~c); }
template <typename I> select(I a, I b, I c) { return detail_::msb_of(c) ? a : b; }

namespace detail_ {
// Note: The OpenCL standard requires isnan() to be available only for floating-point
// types, but some implementations above benefit from it being available for any type
template <typename T> bool isnan(T x) noexcept { return false; }
} // namespace detail_

// relational functions for double arguments

// Provided by CUDA: isfinite, isinf, isnan

int isnormal(double x) noexcept
{
    detail_::destructured_double d = x;
    return isfinite(x) &&  (x == 0 or d.parts.exponent != 0);
}
int isordered(double x, double y) noexcept { return x == x and y == y; }
int isunordered(double x, double y) noexcept { return isnan(x) || isnan(y); }
int signbit(double x) noexcept
{
    detail_::destructured_double d = x;
    return d.parts.sign;
}

// relational functions for float arguments

// Provided by CUDA: isfinite, isinf, isnan

int isnormal(float x) noexcept
{
    detail_::destructured_float d = x;
    return isfinite(x) &&  (x == 0 or d.parts.exponent != 0);
}
int isordered(float x, float y) { return x == x and y == y; }
int isunordered(float x, float y) { return isnan(x) || isnan(y); }
int signbit(float x) noexcept
{
    detail_::destructured_float d = x;
    return d.parts.sign;
}

#ifdef PORT_FROM_OPENCL_ENABLE_HALF_PRECISION
int isfinite(half x) noexcept
{
    detail_::destructured_half d = x;
    return x.parts.exponent != 0b11111;
}
int isinf(half x) noexcept { return not __hisinf(x); }
bool isnan(half x) noexcept { return x == CUDART_NAN_FP16; }
int isordered(half x, half y) noexcept { return x == x and y == y; }
int isunordered(half x, half y) noexcept { return isnan(x) || isnan(y); }
int signbit(half x) noexcept
{
    detail_::destructured_half d = x;
    return d.parts.sign;
}
#endif

// §6.15.7. Vector Data Load and Store Functions
// --------------------------------------------------------------

// Note: Most of these functions are vectorized, and in this file
// we only work on scalars; see the vectorized/ subdirectory

#ifdef PORT_FROM_OPENCL_ENABLE_HALF_PRECISION
double vload_half(size_t offset, const half *p) { return __half2double(*(p + offset)); }
void vstore_half(double data, size_t offset, half *p)      { *(p + offset) = __double2half(data); }
void vstore_half_rte(double data, size_t offset, half *p)  { *(p + offset) = __double2half_rn(data); }
void vstore_half_rtz(double data, size_t offset, half *p)  { *(p + offset) = __double2half_rz(data); }
void vstore_half_rtp(double data, size_t offset, half *p)  { *(p + offset) = __double2half_ru(data); }
void vstore_half_rtn(double data, size_t offset, half *p)  { *(p + offset) = __double2half_rd(data); }

float vload_half(size_t offset, const half *p) { return __half2float(*(p + offset)); }
void vstore_half(float data, size_t offset, half *p)      { *(p + offset) = __float2half(data); }
void vstore_half_rte(float data, size_t offset, half *p)  { *(p + offset) = __float2half_rn(data); }
void vstore_half_rtz(float data, size_t offset, half *p)  { *(p + offset) = __float2half_rz(data); }
void vstore_half_rtp(float data, size_t offset, half *p)  { *(p + offset) = __float2half_ru(data); }
void vstore_half_rtn(float data, size_t offset, half *p)  { *(p + offset) = __float2half_rd(data); }
#endif // PORT_FROM_OPENCL_ENABLE_HALF_PRECISION

// §6.15.8. Synchronization Functions
// --------------------------------------------------------------

typedef enum {
    CLK_LOCAL_MEM_FENCE,
    CLK_GLOBAL_MEM_FENCE,
    CLK_IMAGE_MEM_FENCE // CUDA doesn't really support this AFAICT
} cl_mem_fence_flags;

void work_group_barrier(cl_mem_fence_flags flags)
{
    __syncthreads();
}
void barrier(cl_mem_fence_flags flags) { return work_group_barrier(flags); }
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

// §6.15.14. printf
// --------------------------------------------------------------

// §6.15.15. Image Read and Write Functions
// --------------------------------------------------------------

// §6.15.16. Work-group Collective Functions
// --------------------------------------------------------------

// §6.15.17. Work-group Collective Uniform Arithmetic Functions
// --------------------------------------------------------------

// §6.15.18. Pipe Functions - pipes in general are not supported in CUDA

// §6.15.19. Enqueuing Kernels - not supported, because OpenCL
// in-kernel enqueue requires individual work items to enqueue
// blocks of the additional kernel; CUDA in-kernel has each
// enqueueing thread be responsible for a full launch grid.
// See @url https://developer.download.nvidia.com/assets/cuda/docs/TechBrief_Dynamic_Parallelism_in_CUDA_v2.pdf
// for details on launching kernels-within-kernels in CUDA.



// §6.15.20. Sub-Group Functions - CUDA doesn't support subgroups,
// so skipping these

// §6.15.21. Kernel Clock Functions
// --------------------------------------------------------------

ulong clock_read_device()     { return clock64();             }
ulong clock_read_work_group() { return clock_read_device();   }
ulong clock_read_sub_group()  { return clock_read_sub_group();}

namespace detail_ {
using destructured_ulong = struct {
    ulong value;
    uint2 components;
};
}

uint2 clock_read_hilo_device()
{
    detail_::destructured_ulong clk;
    clk.value = clock64();
    return clk.components;
}
uint2 clock_read_hilo_work_group() { return clock_read_hilo_device(); }
uint2 clock_read_hilo_sub_group()  { return clock_read_hilo_device(); }

#endif // __OPENCL_VERSION__
#endif // PORT_FROM_OPENCL_BUILTIN_FUNCTIONS_CUH_
