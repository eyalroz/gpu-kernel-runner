/**
 * @file
 *
 * @note GE CONFIDENTIAL
 *
 * @author Eyal Rozenberg <Eyal.Rozenberg@GE.com>, SSO 503156516.
 *
 * @brief Definition of the @ref half4 type for CUDA code, utilizing
 * the @ref half and @ref half2 types provided by CUDA headers
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
#ifndef HALF_4_CUH
#define HALF_4_CUH

#include <cuda_fp16.h>

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

// Note: Assuming sizeof(long unsigned int) is 64,
// just like cuda_fp16.h assumes sizeof(unsigned int) is 32

#define __HALF4_TO_ULI(var) *(reinterpret_cast<unsigned long int *>(&(var)))
#define __HALF4_TO_CULI(var) *(reinterpret_cast<const unsigned long int *>(&(var)))

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
} __half4_raw;

//typedef struct __CUDA_ALIGN__(8) {
//    __half2 half2s[2];
//} half2_2;

//#define __HALF4_AS_HALF2S(var) *(reinterpret_cast<half2_2 const *>(&(var)))


/* __half4 is visible to non-nvcc host compilers */
struct __CUDA_ALIGN__(8) __half4 {
    __half x;
    __half y;
    __half z;
    __half w;

    // All construct/copy/assign/move
public:
#if defined(__CPP_VERSION_AT_LEAST_11_FP16)
    __half4() = default;
    __host__ __device__ __half4(const __half4 &&src) { __HALF4_TO_ULI(*this) = std::move(__HALF4_TO_CULI(src)); }
    __host__ __device__ __half4 &operator=(const __half4 &&src) { __HALF4_TO_ULI(*this) = std::move(__HALF4_TO_CULI(src)); return *this; }
#else
    __host__ __device__ __half4() { }
#endif /* defined(__CPP_VERSION_AT_LEAST_11_FP16) */
    __host__ __device__ __half4(const __half &a, const __half &b, const __half &c, const __half &d) : x(a), y(b), z(c), w(d) { }
    __host__ __device__ __half4(const __half2 &ab, const __half2 &cd)
    {
        __half2 * halves = reinterpret_cast<__half2 *>(this);
        halves[0] = ab;
        halves[1] = cd;
    }
    __host__ __device__ __half4(const __half4 &src) { __HALF4_TO_ULI(*this) = __HALF4_TO_CULI(src); }
    __host__ __device__ __half4 &operator=(const __half4 &src) { __HALF4_TO_ULI(*this) = __HALF4_TO_CULI(src); return *this; }

    /* Convert to/from __half4_raw */
    __host__ __device__ __half4(const __half4_raw &h2r ) { __HALF4_TO_ULI(*this) = __HALF4_TO_CULI(h2r); }
    __host__ __device__ __half4 &operator=(const __half4_raw &h2r) { __HALF4_TO_ULI(*this) = __HALF4_TO_CULI(h2r); return *this; }
    __host__ __device__ operator __half4_raw() const {
        __half4_raw ret;
        ret.x = 0U; ret.y = 0U;
        ret.z = 0U; ret.w = 0U; 
        __HALF4_TO_ULI(ret) = __HALF4_TO_CULI(*this); return ret; 
    }
}; // __half4

typedef __half4 half4;

#if defined(__CPP_VERSION_AT_LEAST_11_FP16)
#undef __CPP_VERSION_AT_LEAST_11_FP16
#endif /* defined(__CPP_VERSION_AT_LEAST_11_FP16) */

__host__ __device__ const __half2 * as_halves(const __half4& h4)
{
     return reinterpret_cast<const __half2 *>(&(h4));
}

__host__ __device__ float4 __half42float4(const __half4 h4)
{
    const __half2* halves = as_halves(h4);
    float2 xy = __half22float2(halves[0]);
    float2 yz = __half22float2(halves[1]);
    return { xy.x, xy.y, yz.x, yz.y };
}
//
//__host__ __device__ float4 __float42half4(const __float4 f4)


inline half2 operator+(half lhs, half2 rhs) noexcept { return half2{lhs,lhs} + rhs; }
inline half2 operator-(half lhs, half2 rhs) noexcept { return half2{lhs,lhs} - rhs; }
inline half2 operator*(half lhs, half2 rhs) noexcept { return half2{lhs,lhs} * rhs; }
inline half2 operator/(half lhs, half2 rhs) noexcept { return half2{lhs,lhs} / rhs; }

// half2 with half

inline half2 operator+(half2 lhs, half rhs) noexcept { return lhs + half2{rhs,rhs}; }
inline half2 operator-(half2 lhs, half rhs) noexcept { return lhs - half2{rhs,rhs}; }
inline half2 operator*(half2 lhs, half rhs) noexcept { return lhs * half2{rhs,rhs}; }
inline half2 operator/(half2 lhs, half rhs) noexcept { return lhs / half2{rhs,rhs}; }

inline half2& operator+=(half2& lhs, half rhs) noexcept { lhs = lhs + rhs; return lhs; }
inline half2& operator-=(half2& lhs, half rhs) noexcept { lhs = lhs - rhs; return lhs; }


// half4 with half4

inline half4 operator+(half4 lhs, half4 rhs) noexcept
{
    const half2 *lhs_halves = as_halves(lhs);
    const half2 *rhs_halves = as_halves(rhs);
    return { lhs_halves[0] + rhs_halves[0], lhs_halves[1] + rhs_halves[1] };
}
inline half4 operator-(half4 lhs, half4 rhs) noexcept
{
    const half2 *lhs_halves = as_halves(lhs);
    const half2 *rhs_halves = as_halves(rhs);
    return { lhs_halves[0] - rhs_halves[0], lhs_halves[1] - rhs_halves[1] };
}

inline half4 operator*(half4 lhs, half4 rhs) noexcept
{
    const half2 *lhs_halves = as_halves(lhs);
    const half2 *rhs_halves = as_halves(rhs);
    return { lhs_halves[0] * rhs_halves[0], lhs_halves[1] * rhs_halves[1] };
}
inline half4 operator/(half4 lhs, half4 rhs) noexcept
{
    const half2 *lhs_halves = as_halves(lhs);
    const half2 *rhs_halves = as_halves(rhs);
    return { lhs_halves[0] / rhs_halves[0], lhs_halves[1] / rhs_halves[1] };
}

inline half4& operator+=(half4& lhs, half4 rhs) noexcept { lhs = lhs + rhs; return lhs; }
inline half4& operator-=(half4& lhs, half4 rhs) noexcept { lhs = lhs + rhs; return lhs; }

inline bool operator==(half4 lhs, half4 rhs) noexcept
{
    const half2 *lhs_halves = as_halves(lhs);
    const half2 *rhs_halves = as_halves(rhs);
    return (lhs_halves[0] == rhs_halves[0]) and (lhs_halves[1] == rhs_halves[1]);
};

inline bool operator!=(half4 lhs, half4 rhs) noexcept
{
    const half2 *lhs_halves = as_halves(lhs);
    const half2 *rhs_halves = as_halves(rhs);
    return (lhs_halves[0] != rhs_halves[0]) or (lhs_halves[1] != rhs_halves[1]);
};

// half with half4

inline half4 operator+(half lhs, half4 rhs) noexcept
{
    const half2 * rhs_halves = as_halves(rhs);
    return { lhs + rhs_halves[0], lhs + rhs_halves[1] };
}
inline half4 operator-(half lhs, half4 rhs) noexcept
{
    const half2 * rhs_halves = as_halves(rhs);
    return { lhs - rhs_halves[0], lhs - rhs_halves[1] };
}
inline half4 operator*(half lhs, half4 rhs) noexcept
{
    const half2 * rhs_halves = as_halves(rhs);
    return { lhs * rhs_halves[0], lhs * rhs_halves[1] };
}
inline half4 operator/(half lhs, half4 rhs) noexcept
{
    const half2 * rhs_halves = as_halves(rhs);
    return { lhs / rhs_halves[0], lhs / rhs_halves[1] };
}

// half4 with half

inline half4 operator+(half4 lhs, half rhs) noexcept
{
    const half2 * lhs_halves = as_halves(lhs);
    return { lhs_halves[0] + rhs , lhs_halves[1] + rhs};
}
inline half4 operator-(half4 lhs, half rhs) noexcept
{
    const half2 * lhs_halves = as_halves(lhs);
    return { lhs_halves[0] - rhs , lhs_halves[1] - rhs};
}
inline half4 operator*(half4 lhs, half rhs) noexcept
{
    const half2 * lhs_halves = as_halves(lhs);
    return { lhs_halves[0] * rhs , lhs_halves[1] * rhs};
}
inline half4 operator/(half4 lhs, half rhs) noexcept
{
    const half2 * lhs_halves = as_halves(lhs);
    return { lhs_halves[0] / rhs , lhs_halves[1] / rhs};
}

inline half4& operator+=(half4& lhs, half rhs) noexcept { lhs = lhs + rhs; return lhs; }
inline half4& operator-=(half4& lhs, half rhs) noexcept { lhs = lhs + rhs; return lhs; }


#undef __CUDA_ALIGN__
#undef __HALF4_TO_ULI
#undef __HALF4_TO_CULI

#endif // HALF_4_CUH

