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
 * @note Missing implementations for vector sizes 8 and 16
 *
 */
#ifndef PORT_FROM_OPENCL_VECTORIZED_BUILTIN_FUNCTIONS_CUH_
#define PORT_FROM_OPENCL_VECTORIZED_BUILTIN_FUNCTIONS_CUH_

#ifndef __OPENCL_VERSION__

#include "../opencl_builtin_functions.cuh"

#include <vector_types.h>

// §6.4.3 Explicit conversions
// ===========================

/**
 * The forms of explicit conversion functions for vectorized types with length n:
 *
 *   destTypen convert_destTypen<_sat><_roundingMode>(sourceTypen)
 *
 * ... and the basic types to/from we can convert are:
 * char, uchar, short, ushort, int, uint, long, ulong, float
 *
 * To avoid massive repetitions, we'll be using macros and templates to define
 * the different conversion functions
 */

#define PORT_FROM_OPENCL_DEFINE_FLOAT_TO_INT_CONVERSION_VECTORIZATIONS(tgt, ocl_mode) \
template <typename src> \
inline tgt ## 2 convert_ ## tgt ## 2_ ## ocl_mode(src ## 2 v) noexcept \
{ return { convert_ ## tgt_ ## ocl_mode(v.x), convert_ ## tgt_ ## ocl_mode(v.y) }; } \
template <typename src> \
inline tgt ## 3 convert_ ## tgt ## 3_ ## ocl_mode(src ## 3 v) noexcept \
{ return { convert_ ## tgt_ ## ocl_mode(v.x), convert_ ## tgt_ ## ocl_mode(v.y), convert_ ## tgt_ ## ocl_mode(v.z) }; } \
template <typename src> \
inline tgt ## 4 convert_ ## tgt ## 4_ ## ocl_mode(src ## 4 v) noexcept \
{ return { convert_ ## tgt_ ## ocl_mode(v.x), convert_ ## tgt_ ## ocl_mode(v.y), convert_ ## tgt_ ## ocl_mode(v.z), convert_ ## tgt_ ## ocl_mode(v.w) }; }

#define PORT_FROM_OPENCL_DEFINE_VECTORIZED_CONVERSIONS_TO_INT_TYPE(tgt) \
template <typename src> \
inline tgt convert_ ## tgt ## 2(src ## 2 v) noexcept \
{ return { convert_ ## tgt(v.x), convert_ ## tgt(v.y) }; } \
template <typename src> \
inline tgt convert_ ## tgt ## 3(src ## 3 v) noexcept \
{ return { convert_ ## tgt(v.x), convert_ ## tgt(v.y), convert_ ## tgt(v.z) }; } \
template <typename src> \
inline tgt convert_ ## tgt ## 4(src ## 4 v) noexcept \
{ return { convert_ ## tgt(v.x), convert_ ## tgt(v.y), convert_ ## tgt(v.z), convert_ ## tgt(v.w) }; } \
PORT_FROM_OPENCL_DEFINE_FLOAT_TO_INT_CONVERSION_VECTORIZATIONS(tgt, rtz) \
PORT_FROM_OPENCL_DEFINE_FLOAT_TO_INT_CONVERSION_VECTORIZATIONS(tgt, rte) \
PORT_FROM_OPENCL_DEFINE_FLOAT_TO_INT_CONVERSION_VECTORIZATIONS(tgt, rtp) \
PORT_FROM_OPENCL_DEFINE_FLOAT_TO_INT_CONVERSION_VECTORIZATIONS(tgt, rtn) 

// TODO: Define _sat conversions

PORT_FROM_OPENCL_DEFINE_VECTORIZED_CONVERSIONS_TO_INT_TYPE(char)
PORT_FROM_OPENCL_DEFINE_VECTORIZED_CONVERSIONS_TO_INT_TYPE(uchar)
PORT_FROM_OPENCL_DEFINE_VECTORIZED_CONVERSIONS_TO_INT_TYPE(short)
PORT_FROM_OPENCL_DEFINE_VECTORIZED_CONVERSIONS_TO_INT_TYPE(ushort)
PORT_FROM_OPENCL_DEFINE_VECTORIZED_CONVERSIONS_TO_INT_TYPE(int)
PORT_FROM_OPENCL_DEFINE_VECTORIZED_CONVERSIONS_TO_INT_TYPE(uint)
PORT_FROM_OPENCL_DEFINE_VECTORIZED_CONVERSIONS_TO_INT_TYPE(long)
PORT_FROM_OPENCL_DEFINE_VECTORIZED_CONVERSIONS_TO_INT_TYPE(ulong)

#undef PORT_FROM_OPENCL_DEFINE_VECTORIZED_CONVERSIONS_TO_INT_TYPE
#undef PORT_FROM_OPENCL_DEFINE_FLOAT_TO_INT_CONVERSION_VECTORIZATIONS

// do saturated versions


// §6.4.4.2. Reinterpreting Types Using as_type() and as_typen()
// =============================================================

// "All data types described in Built-in Scalar Data Types and
// Built-in Vector Data Types (except bool, void, and half [19])
// may be also reinterpreted as another data type of the same size"

#define PORT_FROM_OPENCL_DEFINE_VECTORIZED_ASTYPE(tgt) \

template <typename src2> \
inline tgt ## 2 as_ ## tgt ## 2 (src2 v) \
{ \
    static_assert(sizeof(tgt ## 2) == sizeof(src2), "as_type for types of different size is not supported"); \
    return { as_ ## tgt (v.x), as ## tgt (v.y) }; \
}\
template <typename src2> \
inline tgt ## 3 as_ ## tgt ## 3 (src3 v) \
{ \
    static_assert(sizeof(tgt ## 3) == sizeof(src3), "as_type for types of different size is not supported"); \
    return { as_ ## tgt (v.x), as ## tgt (v.y), as ## tgt (v.z) }; \
}\
template <typename src2> \
inline tgt ## 4 as_ ## tgt ## 4 (src4 v) \
{ \
    static_assert(sizeof(tgt ## 4) == sizeof(src4), "as_type for types of different size is not supported"); \
    return { as_ ## tgt (v.x), as ## tgt (v.y), as ## tgt (v.z), as ## tgt (v.w) }; \
}\

PORT_FROM_OPENCL_DEFINE_VECTORIZED_ASTYPE(char)
PORT_FROM_OPENCL_DEFINE_VECTORIZED_ASTYPE(uchar)
PORT_FROM_OPENCL_DEFINE_VECTORIZED_ASTYPE(short)
PORT_FROM_OPENCL_DEFINE_VECTORIZED_ASTYPE(ushort)
PORT_FROM_OPENCL_DEFINE_VECTORIZED_ASTYPE(int)
PORT_FROM_OPENCL_DEFINE_VECTORIZED_ASTYPE(uint)
PORT_FROM_OPENCL_DEFINE_VECTORIZED_ASTYPE(long)
PORT_FROM_OPENCL_DEFINE_VECTORIZED_ASTYPE(ulong)
// half is _not_ supported for astyle
PORT_FROM_OPENCL_DEFINE_VECTORIZED_ASTYPE(float)
PORT_FROM_OPENCL_DEFINE_VECTORIZED_ASTYPE(double)

#undef PORT_FROM_OPENCL_DEFINE_VECTORIZED_ASTYPE

// TODO: Allow for interpreting 4-component vector-types as 3-component OpenCL types, see §6.4.4.2

// §6.15.2. Math Functions
// =======================


#define PORT_FROM_OPENCL_BASIC_VECTORIZATION(f, t1) \
inline t1 ## 2 f(t1 ## 2 v) noexcept { return { f(v.x), f(v.y); }; } \
inline t1 ## 3 f(t1 ## 3 v) noexcept { return { f(v.x), f(v.y), f(v.z); }; } \
inline t1 ## 4 f(t1 ## 4 v) noexcept { return { f(v.x), f(v.y), f(v.z), f(v.w); }; }


#define PORT_FROM_OPENCL_VECTORIZATION_2ARG(f, t1, t2) \
inline t1 ## 2 f(t1 ## 2 v, t2 u) noexcept { return { f(v.x, u), f(v.y, u); }; } \
inline t1 ## 3 f(t1 ## 3 v, t2 u) noexcept { return { f(v.x, u), f(v.y, u), f(v.z, u); }; } \
inline t1 ## 4 f(t1 ## 4 v, t2 u) noexcept { return { f(v.x, u), f(v.y, u), f(v.z, u), f(v.w, u); }; }

#define PORT_FROM_OPENCL_VECTORIZATION_2VEC(f, t1, t2) \
inline t1 ## 2 f(t1 ## 2 v, t2 ## 2 u) noexcept { return { f(v.x, u.x), f(v.y, u.y); }; } \
inline t1 ## 3 f(t1 ## 3 v, t2 ## 3 u) noexcept { return { f(v.x, u.x), f(v.y, u.y), f(v.z, u.z); }; } \
inline t1 ## 4 f(t1 ## 4 v, t2 ## 4 u) noexcept { return { f(v.x, u.x), f(v.y, u.y), f(v.z, u.z), f(v.w, u.w); }; }

#define PORT_FROM_OPENCL_VECTORIZATION_2VECPTR(f, t1, t2) \
inline t1 ## 2 f(t1 ## 2 v, t2 ## 2* u) noexcept { return { f(v.x, &(u->x)), f(v.y, &(u->y)); }; } \
inline t1 ## 3 f(t1 ## 3 v, t2 ## 3* u) noexcept { return { f(v.x, &(u->x)), f(v.y, &(u->y)), f(v.z, &(u->z)); }; } \
inline t1 ## 4 f(t1 ## 4 v, t2 ## 4* u) noexcept { return { f(v.x, &(u->x)), f(v.y, &(u->y)), f(v.z, &(u->z)), f(v.w, &(u->w)); }; }


inline float2 frexp(float2 x, int2 *exp) noexcept { return { frexp(x.x, &exp->x), frexp(x.y, &exp->y) }; }

// Note: can't defined the following to also cover the half-precision type
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(f) \
PORT_FROM_OPENCL_BASIC_VECTORIZATION(f, float) \
PORT_FROM_OPENCL_BASIC_VECTORIZATION(f, double)

// Note: can't defined the following to also cover the half-precision type
PORT_FROM_OPENCL_VECTORIZATION_2ARG_ALL_TYPES(f, t2) \
PORT_FROM_OPENCL_VECTORIZATION_2ARG(f, float, t2) \
PORT_FROM_OPENCL_VECTORIZATION_2ARG(f, double, t2)

// Note: can't defined the following to also cover the half-precision type
PORT_FROM_OPENCL_VECTORIZATION_2VEC_ALL_TYPES(f, t2) \
PORT_FROM_OPENCL_VECTORIZATION_2VEC(f, float, t2) \
PORT_FROM_OPENCL_VECTORIZATION_2VEC(f, double, t2)

// Note: can't defined the following to also cover the half-precision type
PORT_FROM_OPENCL_VECTORIZATION_2VECPTR_ALL_TYPES(f, t2) \
PORT_FROM_OPENCL_VECTORIZATION_2VECPTR(f, float, t2) \
PORT_FROM_OPENCL_VECTORIZATION_2VECPTR(f, double, t2)


inline double2 remquo(double2 x, double2 y, int2 *quo) noexcept
{
    return { remquo(x.x, y.x, &quo->x), remquo(x.y, y.y, &quo->y) };
}

inline double3 remquo(double3 x, double3 y, int3 *quo) noexcept
{
    return {
        remquo(x.x, y.x, &quo->x),
        remquo(x.y, y.y, &quo->y),
        remquo(x.z, y.z, &quo->z)
    };
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

inline double2 frexp(double2 x, int2 *exp) noexcept
{
    return {
        frexp(x.x, &exp->x),
        frexp(x.y, &exp->y),
    };
}

inline double3 frexp(double3 x, int3 *exp) noexcept
{
    return {
        frexp(x.x, &exp->x),
        frexp(x.y, &exp->y),
        frexp(x.z, &exp->z)
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

PORT_FROM_OPENCL_VECTORIZATION_2VEC_ALLTYPES(ldexp, int)
PORT_FROM_OPENCL_VECTORIZATION_2VEC_ALLTYPES(pow, int)
PORT_FROM_OPENCL_VECTORIZATION_2VEC_ALLTYPES(pow, float)
PORT_FROM_OPENCL_VECTORIZATION_2VEC_ALLTYPES(pow, double)
// TODO: More type combinations for pow? But need to make sure the non-vectorized versions exist
PORT_FROM_OPENCL_VECTORIZATION_2VECPTR_ALLTYPES(lgamma_r, int)
PORT_FROM_OPENCL_VECTORIZATION_2VECPTR_ALLTYPES(frexp, int)

// TODO: maxmag
// TODO: minmag
// TODO: mad

// float parameter
// ---------------


PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(acos)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(acosh)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(acospi)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(asin)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(asinh)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(asinpi)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(atan)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(atan2)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(atanh)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(atanpi)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(atan2pi)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(cbrt)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(ceil)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(copysign)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(cos)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(cosh)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(erfc)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(erf)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(exp)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(exp2)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(exp10)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(expm1)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(fabs)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(fdim)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(floor)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(fma)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(fmax)
// Not implemented: gentyped fmax(gentyped x, float y)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(fmin)
// Not implemented: gentyped fmin(gentyped x, float y)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(fmod)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(fract)

PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(log)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(log2)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(log10)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(log1p)
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(logb)

// TODO: implement frexp for floatn with n = 3,8,16
PORT_FROM_OPENCL_BASIC_VECTORIZATION_ALL_TYPES(ilogb)


inline float modf(float x, float *iptr) noexcept { return modff(x, iptr); }
// Not implementing nan, since it doesn't take a parameter which could distinguish float's from doubles etc.
inline float nextafter(float x, float y) noexcept { return nextafterf(x, y); }


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

// half-precision parameter
// -------------------------

// Note: CUDA (as of 13.1) does not provide half-precision versions of most of the
// math functions it offers for float- and double-precision. Also, some OpenCL functions
// which, for double and float, we have implemented using Pi-related constants -
// but we don't have appropriate definitions for the half-precision versions of these.
// Therefore, the macros for float and double don't also cover half-precision, and here
// we only vectorize some functions

PORT_FROM_OPENCL_BASIC_VECTORIZATION(ceil, half)
PORT_FROM_OPENCL_BASIC_VECTORIZATION(cos, half)
PORT_FROM_OPENCL_BASIC_VECTORIZATION(exp, half)
PORT_FROM_OPENCL_BASIC_VECTORIZATION(exp2, half)
PORT_FROM_OPENCL_BASIC_VECTORIZATION(exp10, half)
PORT_FROM_OPENCL_BASIC_VECTORIZATION(floor, half)
#ifdef HAVE_HALF_4
PORT_FROM_OPENCL_VECTORIZATION_2VECPTR(frexp, half, int)
#else
inline half2 frexp(half2 x, int2 *exp) noexcept { return { frexp(x.x, &exp->x), frexp(x.y, &exp->y), }; }
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

// TODO: mad24, mul24

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
#endif // PORT_FROM_OPENCL_VECTORIZED_BUILTIN_FUNCTIONS_CUH_
