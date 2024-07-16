/**
 * @file port_from_opencl.cuh
 *
 * @brief OpenCL-flavor definitions for porting OpenCL kernel code to CUDA
 * with fewer changes required.
 *
 * @copyright (c) 2020-2024, GE HealthCare
 * @copyright (c) 2020-2024, Eyal Rozenberg
 *
 * @license BSD 3-clause license; see the `LICENSE` file or
 * @url https://opensource.org/licenses/BSD-3-Clause
 *
 * @note Can be used for writing kernels targeting both CUDA and OpenCL
 * at once (alongside @ref port_from_cuda.cl.h ).
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
#ifndef PORT_FROM_OPENCL_CUH_
#define PORT_FROM_OPENCL_CUH_

#ifndef __OPENCL_VERSION__

#if __cplusplus < 201103L
#error "This file requires compiling using C++11 or later"
#endif

#include "cuda_syntax_for_ide_parser.cuh"

#ifdef GKR_ENABLE_HALF_PRECISION
#include <cuda_fp16.h>
#include "half4.cuh"
#endif

#include <cstdint>
#include <cstddef> // for size_t
#include <climits>
    // We don't really need this here directly,
    // but failure to include it makes NVRTC
    // take another file rather than our NVRTC-safe climits stub


#include <vector_types.h>

// These defined terms are used in OpenCL and not part of the C++ language
#define __global
#define __private
#define __kernel extern "C" __global__
#define restrict __restrict__
#define __restrict __restrict__
// and note __local is missing!

// For porting, the OpenCL kernel should replace __local
// with one of the following - to indicate which uses of it require
// decorating with CUDA's __shared__ for per-block memory allocation.
#define __local_array __shared__
#define __local_variable __shared__
#define __local_ptr

// This next definition is for "emulating" constexpr in OpenCL - or
// using the next closest thing - a global `__constant` memory space
// definition: The same syntax can be used in both OpenCL and CUDA,
// with CUDA actually producing `constexpr`, and OpenCL using `__constant`
#define CONSTEXPR_OR_CONSTANT_MEM constexpr const

#define CLK_LOCAL_MEM_FENCE 0

template <typename T>
T asin(const T& x);

using uchar = std::uint8_t;
using ushort = std::uint16_t;
using uint = std::uint32_t;
using ulong = std::uint64_t;

// Note: CUDA guarantees that the sizes of non-unsigned char, short, int and long
// are the same as in OpenCL: 1, 2, 4, 8 bytes respectively.

using std::ptrdiff_t;
using std::intptr_t;
using std::uintptr_t;
using std::size_t;

// Note: These are semantically-unsound implementations, which do
// not assume actual floatn alignment, and may result in a subotimal
// choice of SASS instructions

__device__ inline float2 vload2(size_t offset, const float* p)
{
    return { p[offset], p[offset+1] };
//    return reinterpret_cast<const float2*>(p)[offset];
}

__device__ inline void vstore2(const float2& value, size_t offset, float* p)
{
    p[offset  ] = value.x;
    p[offset+1] = value.y;
//    reinterpret_cast<float2*>(p)[offset] = value;
}

__device__ inline float3 vload3(size_t offset, const float* p)
{
    return { p[offset], p[offset+1], p[offset+2] };
//    return reinterpret_cast<const float3*>(p)[offset];
}

__device__ inline void vstore3(const float3& value, size_t offset, float* p)
{
    p[offset  ] = value.x;
    p[offset+1] = value.y;
    p[offset+2] = value.z;
//    reinterpret_cast<float3*>(p)[offset] = value;
}

__device__ inline float4 vload4(size_t offset, const float* p)
{
    return { p[offset], p[offset+1], p[offset+2], p[offset+3] };
//    return reinterpret_cast<const float4*>(p)[offset];
}

__device__ inline void vstore4(const float4& value, size_t offset, float* p)
{
    p[offset  ] = value.x;
    p[offset+1] = value.y;
    p[offset+2] = value.z;
    p[offset+3] = value.w;
//    reinterpret_cast<float4*>(p)[offset] = value;
}

namespace detail {

__device__ inline unsigned int get_dim3_element(const dim3& d3, int index)
{
    switch (index) {
    case 0:  return d3.x;
    case 1:  return d3.y;
    case 2:
    default: return d3.z;
    }
}

} // namespace detail

__device__ inline unsigned int get_local_id(int dimension_index)
{
    return detail::get_dim3_element(threadIdx, dimension_index);
}

__device__ inline unsigned int get_group_id(int dimension_index)
{
    return detail::get_dim3_element(blockIdx, dimension_index);
}

// TODO: Support for larger-than-2^31 grids
//template <typename Size = size_t>
__device__ inline size_t get_global_id(int dimension_index)
{
    // Note: We could have used:
    //
    //  return detail::get_dim3_element(threadIdx, dimension_index) +
    //  detail::get_dim3_element(blockIdx, dimension_index) *
    //  detail::get_dim3_element(blockDim, dimension_index);
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

__device__ inline unsigned int get_local_size(unsigned dimension_index)
{
    return detail::get_dim3_element(blockDim, dimension_index);
}

__device__ inline unsigned int get_num_groups(unsigned dimension_index)
{
    return detail::get_dim3_element(gridDim, dimension_index);
}

__device__ inline size_t get_global_size(unsigned dimension_index)
{
    return static_cast<size_t>(get_num_groups(dimension_index)) * get_local_size(dimension_index);
}

__device__ inline void barrier(int kind)
{
//    assert(kind == CLK_LOCAL_MEM_FENCE);
    __syncthreads();
}

template <typename T>
constexpr __device__ inline unsigned int convert_uint(const T& x) { return static_cast<unsigned int>(x); }

constexpr __device__ inline int2 convert_int2(const float2& v)
{
    return {
        static_cast<int>(v.x),
        static_cast<int>(v.y)
    };
}

constexpr __device__ inline float2 floor(const float2& v) { return { floorf(v.x), floorf(v.y) }; }
constexpr __device__ inline float4 floor(const float4& v)
{
    return { floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w) };
}

template <typename T> constexpr __device__ inline int   convert_int  (const T& x) { return static_cast<int>(x);   }
template <typename T> constexpr __device__ inline float convert_float(const T& x) { return static_cast<float>(x); }

__device__ inline float  native_recip(float  x) { return __frcp_rn(x); }
__device__ inline double native_recip(double x) { return __drcp_rn(x); }

__device__ inline float  native_sqrt(float x)   { return sqrtf(x); }
__device__ inline double native_sqrt(double x)  { return sqrt(x);  }

__device__ inline float  native_rsqrt(float x)  { return rsqrtf(x); }
__device__ inline double native_rsqrt(double x) { return rsqrt(x);  }

//template <typename T, typename Selector>
//T select(T on_false, T on_true, Selector selector);

template <typename T>
struct is_opencl_vectorized { static constexpr const bool value = false; };

template <> struct is_opencl_vectorized<int4> { static constexpr const bool value = true; };
template <> struct is_opencl_vectorized<uint4> { static constexpr const bool value = true; };
template <> struct is_opencl_vectorized<float4> { static constexpr const bool value = true; };
template <> struct is_opencl_vectorized<int2> { static constexpr const bool value = true; };
template <> struct is_opencl_vectorized<uint2> { static constexpr const bool value = true; };
template <> struct is_opencl_vectorized<float2> { static constexpr const bool value = true; };
// TODO: Fill in more vector types

template <typename Scalar>
__device__ inline Scalar select(
    Scalar on_false,
    Scalar on_true,
    int selector)
{
	static_assert(is_opencl_vectorized<Scalar>::value == false, "Don't use this on vector types");
    return selector ? on_true : on_false;
}

// Arithmetic and assignment operators for vectorized types

// short2 with short2

constexpr __device__ inline short2 operator+(short2 lhs, short2 rhs) noexcept { return { (short) (lhs.x + rhs.x), (short) (lhs.y + rhs.y) }; }
constexpr __device__ inline short2 operator-(short2 lhs, short2 rhs) noexcept { return { (short) (lhs.x - rhs.x), (short) (lhs.y - rhs.y) }; }
constexpr __device__ inline short2 operator*(short2 lhs, short2 rhs) noexcept { return { (short) (lhs.x * rhs.x), (short) (lhs.y * rhs.y) }; }
constexpr __device__ inline short2 operator/(short2 lhs, short2 rhs) noexcept { return { (short) (lhs.x / rhs.x), (short) (lhs.y / rhs.y) }; }

constexpr __device__ inline short2& operator+=(short2& lhs, short2 rhs) noexcept { lhs = lhs + rhs; return lhs; }
constexpr __device__ inline short2& operator-=(short2& lhs, short2 rhs) noexcept { lhs = lhs - rhs; return lhs; }

// short with short2

constexpr __device__ inline short2 operator+(short lhs, short2 rhs) noexcept { return { (short) (lhs + rhs.x), (short) (lhs + rhs.y) }; }
constexpr __device__ inline short2 operator-(short lhs, short2 rhs) noexcept { return { (short) (lhs - rhs.x), (short) (lhs - rhs.y) }; }
constexpr __device__ inline short2 operator*(short lhs, short2 rhs) noexcept { return { (short) (lhs * rhs.x), (short) (lhs * rhs.y) }; }
constexpr __device__ inline short2 operator/(short lhs, short2 rhs) noexcept { return { (short) (lhs / rhs.x), (short) (lhs / rhs.y) }; }

// short2 with short

constexpr __device__ inline short2 operator+(short2 lhs, short rhs) noexcept { return { (short) (lhs.x + rhs), (short) (lhs.y + rhs) }; }
constexpr __device__ inline short2 operator-(short2 lhs, short rhs) noexcept { return { (short) (lhs.x - rhs), (short) (lhs.y - rhs) }; }
constexpr __device__ inline short2 operator*(short2 lhs, short rhs) noexcept { return { (short) (lhs.x * rhs), (short) (lhs.y * rhs) }; }
constexpr __device__ inline short2 operator/(short2 lhs, short rhs) noexcept { return { (short) (lhs.x / rhs), (short) (lhs.y / rhs) }; }

constexpr __device__ inline short2& operator+=(short2& lhs, short rhs) noexcept { lhs = lhs + rhs; return lhs; }
constexpr __device__ inline short2& operator-=(short2& lhs, short rhs) noexcept { lhs = lhs - rhs; return lhs; }


// int2 with int2

constexpr __device__ inline int2 operator+(int2 lhs, int2 rhs) noexcept { return { lhs.x + rhs.x, lhs.y + rhs.y }; }
constexpr __device__ inline int2 operator-(int2 lhs, int2 rhs) noexcept { return { lhs.x - rhs.x, lhs.y - rhs.y }; }
constexpr __device__ inline int2 operator*(int2 lhs, int2 rhs) noexcept { return { lhs.x * rhs.x, lhs.y * rhs.y }; }
constexpr __device__ inline int2 operator/(int2 lhs, int2 rhs) noexcept { return { lhs.x / rhs.x, lhs.y / rhs.y }; }

constexpr __device__ inline int2& operator+=(int2& lhs, int2 rhs) noexcept { lhs = lhs + rhs; return lhs; }
constexpr __device__ inline int2& operator-=(int2& lhs, int2 rhs) noexcept { lhs = lhs - rhs; return lhs; }

// int with int2

constexpr __device__ inline int2 operator+(int lhs, int2 rhs) noexcept { return { lhs + rhs.x, lhs + rhs.y }; }
constexpr __device__ inline int2 operator-(int lhs, int2 rhs) noexcept { return { lhs - rhs.x, lhs - rhs.y }; }
constexpr __device__ inline int2 operator*(int lhs, int2 rhs) noexcept { return { lhs * rhs.x, lhs * rhs.y }; }
constexpr __device__ inline int2 operator/(int lhs, int2 rhs) noexcept { return { lhs / rhs.x, lhs / rhs.y }; }

// int2 with int

constexpr __device__ inline int2 operator+(int2 lhs, int rhs) noexcept { return { lhs.x + rhs, lhs.y + rhs }; }
constexpr __device__ inline int2 operator-(int2 lhs, int rhs) noexcept { return { lhs.x - rhs, lhs.y - rhs }; }
constexpr __device__ inline int2 operator*(int2 lhs, int rhs) noexcept { return { lhs.x * rhs, lhs.y * rhs }; }
constexpr __device__ inline int2 operator/(int2 lhs, int rhs) noexcept { return { lhs.x / rhs, lhs.y / rhs }; }

constexpr __device__ inline int2& operator+=(int2& lhs, int rhs) noexcept { lhs = lhs + rhs; return lhs; }
constexpr __device__ inline int2& operator-=(int2& lhs, int rhs) noexcept { lhs = lhs - rhs; return lhs; }

// int4 with int4

__device__ inline int4 operator+(int4 lhs, int4 rhs) noexcept { return { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w }; }
__device__ inline int4 operator-(int4 lhs, int4 rhs) noexcept { return { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w }; }
__device__ inline int4 operator*(int4 lhs, int4 rhs) noexcept { return { lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w }; }
__device__ inline int4 operator/(int4 lhs, int4 rhs) noexcept { return { lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w }; }

__device__ inline int4& operator+=(int4& lhs, int4 rhs) noexcept { lhs = lhs + rhs; return lhs; }
__device__ inline int4& operator-=(int4& lhs, int4 rhs) noexcept { lhs = lhs + rhs; return lhs; }

// int with int4

__device__ inline int4 operator+(int lhs, int4 rhs) noexcept { return { lhs + rhs.x, lhs + rhs.y, lhs + rhs.z, lhs + rhs.w }; }
__device__ inline int4 operator-(int lhs, int4 rhs) noexcept { return { lhs - rhs.x, lhs - rhs.y, lhs - rhs.z, lhs - rhs.w }; }
__device__ inline int4 operator*(int lhs, int4 rhs) noexcept { return { lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w }; }
__device__ inline int4 operator/(int lhs, int4 rhs) noexcept { return { lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.w }; }

// int4 with int

__device__ inline int4 operator+(int4 lhs, int rhs) noexcept { return { lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs }; }
__device__ inline int4 operator-(int4 lhs, int rhs) noexcept { return { lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs }; }
__device__ inline int4 operator*(int4 lhs, int rhs) noexcept { return { lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs }; }
__device__ inline int4 operator/(int4 lhs, int rhs) noexcept { return { lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs }; }

__device__ inline int4& operator+=(int4& lhs, int rhs) noexcept { lhs = lhs + rhs; return lhs; }
__device__ inline int4& operator-=(int4& lhs, int rhs) noexcept { lhs = lhs + rhs; return lhs; }



// float2 with float2

constexpr __device__ inline float2 operator+(float2 lhs, float2 rhs) noexcept { return { lhs.x + rhs.x, lhs.y + rhs.y }; }
constexpr __device__ inline float2 operator-(float2 lhs, float2 rhs) noexcept { return { lhs.x - rhs.x, lhs.y - rhs.y }; }
constexpr __device__ inline float2 operator*(float2 lhs, float2 rhs) noexcept { return { lhs.x * rhs.x, lhs.y * rhs.y }; }
constexpr __device__ inline float2 operator/(float2 lhs, float2 rhs) noexcept { return { lhs.x / rhs.x, lhs.y / rhs.y }; }

constexpr __device__ inline float2& operator+=(float2& lhs, float2 rhs) noexcept { lhs = lhs + rhs; return lhs; }
constexpr __device__ inline float2& operator-=(float2& lhs, float2 rhs) noexcept { lhs = lhs - rhs; return lhs; }

// float with float2

constexpr __device__ inline float2 operator+(float lhs, float2 rhs) noexcept { return { lhs + rhs.x, lhs + rhs.y }; }
constexpr __device__ inline float2 operator-(float lhs, float2 rhs) noexcept { return { lhs - rhs.x, lhs - rhs.y }; }
constexpr __device__ inline float2 operator*(float lhs, float2 rhs) noexcept { return { lhs * rhs.x, lhs * rhs.y }; }
constexpr __device__ inline float2 operator/(float lhs, float2 rhs) noexcept { return { lhs / rhs.x, lhs / rhs.y }; }

// float2 with float

constexpr __device__ inline float2 operator+(float2 lhs, float rhs) noexcept { return { lhs.x + rhs, lhs.y + rhs }; }
constexpr __device__ inline float2 operator-(float2 lhs, float rhs) noexcept { return { lhs.x - rhs, lhs.y - rhs }; }
constexpr __device__ inline float2 operator*(float2 lhs, float rhs) noexcept { return { lhs.x * rhs, lhs.y * rhs }; }
constexpr __device__ inline float2 operator/(float2 lhs, float rhs) noexcept { return { lhs.x / rhs, lhs.y / rhs }; }

constexpr __device__ inline float2& operator+=(float2& lhs, float rhs) noexcept { lhs = lhs + rhs; return lhs; }
constexpr __device__ inline float2& operator-=(float2& lhs, float rhs) noexcept { lhs = lhs - rhs; return lhs; }

// float4 with float4

__device__ inline float4 operator+(float4 lhs, float4 rhs) noexcept { return { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w }; }
__device__ inline float4 operator-(float4 lhs, float4 rhs) noexcept { return { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w }; }
__device__ inline float4 operator*(float4 lhs, float4 rhs) noexcept { return { lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w }; }
__device__ inline float4 operator/(float4 lhs, float4 rhs) noexcept { return { lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w }; }

__device__ inline float4& operator+=(float4& lhs, float4 rhs) noexcept { lhs = lhs + rhs; return lhs; }
__device__ inline float4& operator-=(float4& lhs, float4 rhs) noexcept { lhs = lhs + rhs; return lhs; }

// float with float4

__device__ inline float4 operator+(float lhs, float4 rhs) noexcept { return { lhs + rhs.x, lhs + rhs.y, lhs + rhs.z, lhs + rhs.w }; }
__device__ inline float4 operator-(float lhs, float4 rhs) noexcept { return { lhs - rhs.x, lhs - rhs.y, lhs - rhs.z, lhs - rhs.w }; }
__device__ inline float4 operator*(float lhs, float4 rhs) noexcept { return { lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w }; }
__device__ inline float4 operator/(float lhs, float4 rhs) noexcept { return { lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.w }; }

// float4 with float

__device__ inline float4 operator+(float4 lhs, float rhs) noexcept { return { lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs }; }
__device__ inline float4 operator-(float4 lhs, float rhs) noexcept { return { lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs }; }
__device__ inline float4 operator*(float4 lhs, float rhs) noexcept { return { lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs }; }
__device__ inline float4 operator/(float4 lhs, float rhs) noexcept { return { lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs }; }

__device__ inline float4& operator+=(float4& lhs, float rhs) noexcept { lhs = lhs + rhs; return lhs; }
__device__ inline float4& operator-=(float4& lhs, float rhs) noexcept { lhs = lhs + rhs; return lhs; }

// float4 with array of 4 floats

__device__ inline float4 as_float4(float const(& floats)[4]) noexcept
{
    float4 result;
    result.x = floats[0];
    result.y = floats[1];
    result.z = floats[2];
    result.w = floats[3];
    return result;
}

// array of 4 floats with float4

typedef float float_array4[4];

__device__ inline float_array4& as_float_array(float4& floats) noexcept
{
    return reinterpret_cast<float_array4 &>(floats);
}

__device__ inline float4 operator+(float_array4& lhs, float4 rhs) noexcept { float4 lhs_ = as_float4(lhs); return lhs_ + rhs; }
__device__ inline float4 operator-(float_array4& lhs, float4 rhs) noexcept { float4 lhs_ = as_float4(lhs); return lhs_ - rhs; }
__device__ inline float4 operator*(float_array4& lhs, float4 rhs) noexcept { float4 lhs_ = as_float4(lhs); return lhs_ * rhs; }
__device__ inline float4 operator/(float_array4& lhs, float4 rhs) noexcept { float4 lhs_ = as_float4(lhs); return lhs_ / rhs; }

__device__ inline float_array4& operator+=(float_array4& lhs, float4 rhs) noexcept
{
    lhs[0] += rhs.x;
    lhs[1] += rhs.y;
    lhs[2] += rhs.z;
    lhs[3] += rhs.w;
    return lhs;
}

__device__ inline float_array4& operator-=(float_array4& lhs, float4 rhs) noexcept
{
    lhs[0] -= rhs.x;
    lhs[1] -= rhs.y;
    lhs[2] -= rhs.z;
    lhs[3] -= rhs.w;
    return lhs;
}

// TODO: Add the operators involving float2's and arrays of 2 floats.
// TODO: Add operators for other types, or template all of the above on the scalar type

#ifndef __CLANG_CUDA_MATH_H__
__device__ inline float fdividef (float x, float y ) { return __fdividef(x, y); }
    // Note: We don't need to define fdivide - that's already defined, strangely enough
    // (and __fdivide isn't).
#endif // __CLANG_CUDA_MATH_H__

/**
 * The following macro is intended to allow the same syntax for constructing compound types
 * in both OpenCL and CUDA. In CUDA, we would write float2 { foo, bar }; but in OpenCL we
 * would write that (float2) { foo, bar };
 */
#define make_compound(_compound_type) _compound_type

/*
 * The ternary selection operator (?:) operates on three expressions (exp1 ? exp2 : exp3).
 * This operator evaluates the first expression exp1, which can be a scalar or vector result except float.
 * If all three expressions are scalar values, the C99 rules for ternary operator are followed. If the
 * result is a vector value, then this is equivalent to calling select(exp3, exp2, exp1). The select
 * function is described in Scalar and Vector Relational Functions. The second and third expressions
 * can be any type, as long their types match, or there is an implicit conversion that can be
 * applied to one of the expressions to make their types match, or one is a vector and the
 * other is a scalar and the scalar may be subject to the usual arithmetic conversion to the element
 * type used by the vector operand and widened to the same type as the vector type. This resulting
 * matching type is the type of the entire expression.
*/

#endif // __OPENCL_VERSION__
#endif // PORT_FROM_OPENCL_CUH_
