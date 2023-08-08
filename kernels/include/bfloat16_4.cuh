/**
 * @file
 *
 * @note GE CONFIDENTIAL
 *
 * @author Eyal Rozenberg <Eyal.Rozenberg@GE.com>, SSO 503156516.
 *
 * @brief Definition of the @__bfloat164 class and related constructs
 *
 * @copyright (c) 2023 GE HealthCare, All Rights Reserved
 *
 * @license
 * This unpublished material is proprietary to GE HealthCare. The methods
 * and techniques described herein are considered trade secrets and/or
 * confidential. Reproduction or distribution, in whole or in part, is
 * forbidden except by express written permission of GE HealthCare.
 * GE is a trademark of General Electric Company used under trademark license.
 */
#ifndef BLFOAT16_4_CUH
#define BLFOAT16_4_CUH

#include <cuda_bf16.h>

#define BF16_HD __host__ __device__

#if !defined(_MSC_VER) && __cplusplus >= 201103L
#   define __CPP_VERSION_AT_LEAST_11_FP16
#elif _MSC_FULL_VER >= 190024210 && _MSVC_LANG >= 201103L
#   define __CPP_VERSION_AT_LEAST_11_FP16
#endif

/* C++11 header for std::move.
 * In RTC mode, std::move is provided implicitly; don't include the header
 */
#if defined(__CPP_VERSION_AT_LEAST_11_BF16) && !defined(__CUDACC_RTC__)
#include <utility>
#endif /* defined(__CPP_VERSION_AT_LEAST_11_BF16) && !defined(__CUDACC_RTC__) */


#if !defined(IF_DEVICE_OR_CUDACC)
#if defined(__CUDACC__)
#define IF_DEVICE_OR_CUDACC(d, c, f) NV_IF_ELSE_TARGET(NV_IS_DEVICE, d, c)
#else
#define IF_DEVICE_OR_CUDACC(d, c, f) NV_IF_ELSE_TARGET(NV_IS_DEVICE, d, f)
#endif
#endif

#define __CUDA_ALIGN__(align) __align__(align)

/* Macros to allow nv_bfloat16 & nv_bfloat162 to be used by inline assembly */
//#define __BFLOAT16_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
//#define __BFLOAT16_TO_CUS(var) *(reinterpret_cast<const unsigned short *>(&(var)))
#define __BFLOAT164_TO_ULI(var) *(reinterpret_cast<unsigned long int *>(&(var)))
#define __BFLOAT164_TO_CULI(var) *(reinterpret_cast<const unsigned long int *>(&(var)))

// define __CUDA_BF16_CONSTEXPR__ in order to
// use constexpr where possible, with supporting C++ dialects
// undef after use
#if (defined __CPP_VERSION_AT_LEAST_11_BF16)
#define __CUDA_BF16_CONSTEXPR__   constexpr
#else
#define __CUDA_BF16_CONSTEXPR__
#endif
/* Hide GCC member initialization list warnings because of host/device in-function init requirement */
#if defined(__GNUC__)
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Weffc++"
#endif /* __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6) */
#endif /* defined(__GNUC__) */

/* class' : multiple assignment operators specified
   The class has multiple assignment operators of a single type. This warning is informational */
#if defined(_MSC_VER) && _MSC_VER >= 1500
#pragma warning( push )
#pragma warning( disable:4522 )
#endif /* defined(__GNUC__) */


typedef struct __CUDA_ALIGN__(8) {
    unsigned short x;
    unsigned short y;
    unsigned short z;
    unsigned short w;
} __bfloat164_raw;

/**
* \ingroup CUDA_MATH_INTRINSIC_BFLOAT16
 * \brief nv_bfloat164 datatype
 * \details This structure implements the datatype for storing four
 * nv_bfloat16 floating-point numbers. 
 * The structure implements assignment, arithmetic and comparison
 * operators, and type conversions. 
 * 
 * - NOTE: __nv_bfloat162 is visible to non-nvcc host compilers
 */
struct __CUDA_ALIGN__(8) __nv_bfloat164 {
    ///  Storage fields  holding the individual __nv_bfloat16's
    ///@{
    __nv_bfloat16 x;
    __nv_bfloat16 y;
    __nv_bfloat16 z;
    __nv_bfloat16 w;
    ///@}

    // All construct/copy/assign/move
public: // constructors
#if defined(__CPP_VERSION_AT_LEAST_11_BF16)
    __nv_bfloat164() = default;
    BF16_HD __nv_bfloat164(__nv_bfloat164 &&src) { __BFLOAT164_TO_UI(*this) = std::move(__BFLOAT164_TO_CULI(src)); }
#else
    BF16_HD __nv_bfloat164() { }
#endif /* defined(__CPP_VERSION_AT_LEAST_11_BF16) */
    BF16_HD __CUDA_BF16_CONSTEXPR__ __nv_bfloat164(
        const __nv_bfloat16 &a,
        const __nv_bfloat16 &b,
        const __nv_bfloat16 &c,
        const __nv_bfloat16 &d)
        : x(a), y(b), z(c), w(d) { }
    BF16_HD __nv_bfloat164(const __nv_bfloat162 &ab, const __nv_bfloat162 &cd)
    {
        __nv_bfloat162 * halves = reinterpret_cast<__nv_bfloat162 *>(this);
        halves[0] = ab;
        halves[1] = cd;
    }
    BF16_HD __nv_bfloat164(const __nv_bfloat164 &src) { __BFLOAT164_TO_ULI(*this) = __BFLOAT164_TO_CULI(src); }


public: // operators
    BF16_HD __nv_bfloat164 &operator=(const __nv_bfloat164 &src) { __BFLOAT164_TO_ULI(*this) = __BFLOAT164_TO_CULI(src); return *this; }
#if defined(__CPP_VERSION_AT_LEAST_11_BF16)
    BF16_HD __nv_bfloat164 &operator=(__nv_bfloat164 &&src) { __BFLOAT164_TO_ULI(*this) = std::move(__BFLOAT164_TO_CULI(src)); return *this; }
#endif

public: // Convert to/from __bfloat164_raw

    BF16_HD __nv_bfloat164(const __bfloat164_raw &h2r ) { __BFLOAT164_TO_ULI(*this) = __BFLOAT164_TO_CULI(h2r); }
    BF16_HD __nv_bfloat164 &operator=(const __bfloat164_raw &h2r) { __BFLOAT164_TO_ULI(*this) = __BFLOAT164_TO_CULI(h2r); return *this; }
    BF16_HD operator __bfloat164_raw() const
    {
        __bfloat164_raw ret;
        ret.x = 0U; ret.y = 0U;
        ret.z = 0U; ret.w = 0U;
        __BFLOAT164_TO_ULI(ret) = __BFLOAT164_TO_CULI(*this); return ret;
    }
};

BF16_HD float4 __bfloat1642float4(const __nv_bfloat164 h4)
{
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    const __nv_bfloat162 * halves = reinterpret_cast<const __nv_bfloat162 *>(&(h4));
    float2 xy = __bfloat1622float2(halves[0]);
    float2 yz = __bfloat1622float2(halves[1]);
    return { xy.x, xy.y, yz.x, yz.y };
#else
    return { h4.x, h4.y, h4.z, h4.w };
#endif
}

// TODO:
// BF16_HD float4 __float42bfloat16_4(const __float4 f4)

// __nv_bfloat16 with __nv_bfloat16

inline __nv_bfloat16& operator+=(__nv_bfloat16& lhs, __nv_bfloat16 rhs) noexcept { lhs = lhs + rhs; return lhs; }
inline __nv_bfloat16& operator-=(__nv_bfloat16& lhs, __nv_bfloat16 rhs) noexcept { lhs = lhs - rhs; return lhs; }

// __nv_bfloat16 with __nv_bfloat162

inline __nv_bfloat162 operator+(__nv_bfloat16 lhs, __nv_bfloat162 rhs) noexcept { return { lhs + rhs.x, lhs + rhs.y }; }
inline __nv_bfloat162 operator-(__nv_bfloat16 lhs, __nv_bfloat162 rhs) noexcept { return { lhs - rhs.x, lhs - rhs.y }; }
inline __nv_bfloat162 operator*(__nv_bfloat16 lhs, __nv_bfloat162 rhs) noexcept { return { lhs * rhs.x, lhs * rhs.y }; }
inline __nv_bfloat162 operator/(__nv_bfloat16 lhs, __nv_bfloat162 rhs) noexcept { return { lhs / rhs.x, lhs / rhs.y }; }

// __nv_bfloat162 with __nv_bfloat16

inline __nv_bfloat162 operator+(__nv_bfloat162 lhs, __nv_bfloat16 rhs) noexcept { return { lhs.x + rhs, lhs.y + rhs }; }
inline __nv_bfloat162 operator-(__nv_bfloat162 lhs, __nv_bfloat16 rhs) noexcept { return { lhs.x - rhs, lhs.y - rhs }; }
inline __nv_bfloat162 operator*(__nv_bfloat162 lhs, __nv_bfloat16 rhs) noexcept { return { lhs.x * rhs, lhs.y * rhs }; }
inline __nv_bfloat162 operator/(__nv_bfloat162 lhs, __nv_bfloat16 rhs) noexcept { return { lhs.x / rhs, lhs.y / rhs }; }

inline __nv_bfloat162& operator+=(__nv_bfloat162& lhs, __nv_bfloat16 rhs) noexcept { lhs = lhs + rhs; return lhs; }
inline __nv_bfloat162& operator-=(__nv_bfloat162& lhs, __nv_bfloat16 rhs) noexcept { lhs = lhs - rhs; return lhs; }

// __nv_bfloat164 with __nv_bfloat164

inline __nv_bfloat164 operator+(__nv_bfloat164 lhs, __nv_bfloat164 rhs) noexcept { return { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w }; }
inline __nv_bfloat164 operator-(__nv_bfloat164 lhs, __nv_bfloat164 rhs) noexcept { return { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w }; }
inline __nv_bfloat164 operator*(__nv_bfloat164 lhs, __nv_bfloat164 rhs) noexcept { return { lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w }; }
inline __nv_bfloat164 operator/(__nv_bfloat164 lhs, __nv_bfloat164 rhs) noexcept { return { lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w }; }

inline __nv_bfloat164& operator+=(__nv_bfloat164& lhs, __nv_bfloat164 rhs) noexcept { lhs = lhs + rhs; return lhs; }
inline __nv_bfloat164& operator-=(__nv_bfloat164& lhs, __nv_bfloat164 rhs) noexcept { lhs = lhs + rhs; return lhs; }

// __nv_bfloat16 with __nv_bfloat164

inline __nv_bfloat164 operator+(__nv_bfloat16 lhs, __nv_bfloat164 rhs) noexcept { return { lhs + rhs.x, lhs + rhs.y, lhs + rhs.z, lhs + rhs.w }; }
inline __nv_bfloat164 operator-(__nv_bfloat16 lhs, __nv_bfloat164 rhs) noexcept { return { lhs - rhs.x, lhs - rhs.y, lhs - rhs.z, lhs - rhs.w }; }
inline __nv_bfloat164 operator*(__nv_bfloat16 lhs, __nv_bfloat164 rhs) noexcept { return { lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w }; }
inline __nv_bfloat164 operator/(__nv_bfloat16 lhs, __nv_bfloat164 rhs) noexcept { return { lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.w }; }

// __nv_bfloat164 with __nv_bfloat16

inline __nv_bfloat164 operator+(__nv_bfloat164 lhs, __nv_bfloat16 rhs) noexcept { return { lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs }; }
inline __nv_bfloat164 operator-(__nv_bfloat164 lhs, __nv_bfloat16 rhs) noexcept { return { lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs }; }
inline __nv_bfloat164 operator*(__nv_bfloat164 lhs, __nv_bfloat16 rhs) noexcept { return { lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs }; }
inline __nv_bfloat164 operator/(__nv_bfloat164 lhs, __nv_bfloat16 rhs) noexcept { return { lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs }; }

inline __nv_bfloat164& operator+=(__nv_bfloat164& lhs, __nv_bfloat16 rhs) noexcept { lhs = lhs + rhs; return lhs; }
inline __nv_bfloat164& operator-=(__nv_bfloat164& lhs, __nv_bfloat16 rhs) noexcept { lhs = lhs + rhs; return lhs; }

#if defined(__CPP_VERSION_AT_LEAST_11_FP16)
#undef __CPP_VERSION_AT_LEAST_11_FP16
#endif /* defined(__CPP_VERSION_AT_LEAST_11_FP16) */

#undef __CUDA_ALIGN__
#undef __HALF4_TO_ULI
#undef __HALF4_TO_CULI

#endif // BLFOAT16_4_CUH
