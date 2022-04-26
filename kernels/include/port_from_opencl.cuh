/**
 * @file port_from_opencl.cuh
 *
 * @brief OpenCL-flavor definitions for porting OpenCL kernel code to CUDA
 * with minimal required changes
 */
#ifndef PORT_FROM_OPENCL_CUH_
#define PORT_FROM_OPENCL_CUH_

#ifndef __OPENCL_VERSION__

#include <cstdint>
#include <cstddef> // for size_t
#include <climits>
    // We don't really need this here directly,
    // but failure to include it makes NVRTC
    // take another file rather than our NVRTC-safe climits stub


// Do we need these?
#include <sm_20_intrinsics.h>
// #include <device_functions.h>

#include <vector_types.h>

// #include <cassert>
// #include <type_traits>

#ifdef __CDT_PARSER__
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

#define __device__
#define __device_builtin__

/*
Is there a header which gets us max and min?
 */

#endif // __CDT_PARSER__

// These defined terms are used in OpenCL and not part of the C++ language
#define __global
#define __private
#define __kernel __global__
#define __constant constexpr const
#define restrict __restrict
// and note __local is missing!

// For porting, the OpenCL kernel should replace __local
// with one of the following - to indicate which uses of it require
// decorating with CUDA's __shared__ for per-block memory allocation.
#define __local_array __shared__
#define __local_ptr

#define CLK_LOCAL_MEM_FENCE 0

template <typename T>
T asin(const T& x);

// barrier, select, get_local_id, get_global_id, floor, convert_uint,
// floor, convert_int, convert_float, native_recip

using std::size_t;
using uint = std::uint32_t;
using ushort = std::uint16_t;

inline float2 vload2(size_t offset, const float* p)
{
    return reinterpret_cast<const float2*>(p)[offset];
}

inline void vstore2(const float2& value, size_t offset, float* p)
{
    reinterpret_cast<float2*>(p)[offset] = value;
}

inline float4 vload4(size_t offset, const float* p)
{
    return reinterpret_cast<const float4*>(p)[offset];
}

inline void vstore4(const float4& value, size_t offset, float* p)
{
    reinterpret_cast<float4*>(p)[offset] = value;
}

namespace detail {

inline unsigned get_dim3_element(const dim3& d3, int index)
{
    switch (index) {
    case 0:  return d3.x;
    case 1:  return d3.y;
    case 2:
    default: return d3.z;
    }
}

}

inline uint get_local_id(int dimension_index)
{
    return detail::get_dim3_element(threadIdx, dimension_index);
}

inline uint get_group_id(int dimension_index)
{
    return detail::get_dim3_element(blockIdx, dimension_index);
}

// TODO: Support for larger-than-2^31 grids
//template <typename Size = size_t>
inline size_t get_global_id(int dimension_index)
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

inline unsigned get_local_size(uint dimension_index)
{
    return detail::get_dim3_element(blockDim, dimension_index);
}

inline unsigned get_num_groups(uint dimension_index)
{
    return detail::get_dim3_element(gridDim, dimension_index);
}

inline size_t get_global_size(uint dimension_index)
{
    return static_cast<size_t>(get_num_groups(dimension_index)) * get_local_size(dimension_index);
}


inline void barrier(int kind)
{
//    assert(kind == CLK_LOCAL_MEM_FENCE);
    __syncthreads();
}

template <typename T>
inline uint convert_uint(const T& x) { return static_cast<uint>(x); }

inline int2 convert_int2(const float2& v)
{
    return {
        static_cast<int>(v.x),
        static_cast<int>(v.y)
    };
}

inline float2 floor(const float2& v) { return { floorf(v.x), floorf(v.y) }; }
inline float4 floor(const float4& v)
{
    return { floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w) };
}

template <typename T>
inline int convert_int(const T& x) { return static_cast<int>(x); }

template <typename T>
inline float convert_float(const T& x) { return static_cast<float>(x); }

inline float native_recip(const float x)
{
    return __frcp_rn(x);
}

double native_recip(const double x)
{
    return __drcp_rn(x);
}

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
inline Scalar select(
    Scalar on_false,
    Scalar on_true,
    int selector)
{
	static_assert(is_opencl_vectorized<Scalar>::value == false, "Don't use this on vector types");
    return selector ? on_true : on_false;
}

inline float4 operator*(float lhs, float4 rhs) noexcept
{
    return {
        rhs.x * lhs,
        rhs.y * lhs,
        rhs.z * lhs,
        rhs.w * lhs
    };
}

inline float2 operator*(float lhs, float2 rhs) noexcept
{
    return {
        lhs * rhs.x,
        lhs * rhs.y
    };
}

inline float2 operator-(float2 lhs, float2 rhs) noexcept
{
    return {
        lhs.x - rhs.x,
        lhs.y - rhs.y
    };
}

inline float2 operator-(float2 lhs, uint2 rhs) noexcept
{
    return {
        lhs.x - rhs.x,
        lhs.y - rhs.y
    };
}


inline float2 operator*(float2 lhs, float rhs) noexcept
{
    return {
        lhs.x * rhs,
        lhs.y * rhs
    };
}

inline float4 operator+(float4 lhs, float4 rhs) noexcept
{
    return {
        lhs.x + rhs.x,
        lhs.y + rhs.y,
        lhs.z + rhs.z,
        lhs.w + rhs.w
    };
}

inline float2 operator+(float2 lhs, float2 rhs) noexcept
{
    return {
        lhs.x + rhs.x,
        lhs.y + rhs.y
    };
}

inline float4 operator+=(float4& lhs, float4 rhs) noexcept
{
    return {
        lhs.x += rhs.x,
        lhs.y += rhs.y,
        lhs.z += rhs.z,
        lhs.w += rhs.w
    };
}

inline float4 operator+=(float(& lhs)[4], float4 rhs) noexcept
{
    return {
        lhs[0] += rhs.x,
        lhs[1] += rhs.y,
        lhs[2] += rhs.z,
        lhs[3] += rhs.w
    };
}


inline float2 operator+=(float2& lhs, float4 rhs) noexcept
{
    return {
        lhs.x += rhs.x,
        lhs.y += rhs.y
    };
}

inline float4 operator-(float4 lhs, float rhs) noexcept
{
    return {
        lhs.x - rhs,
        lhs.y - rhs,
        lhs.z - rhs,
        lhs.w - rhs
    };
}

inline float2 operator-(float2 lhs, float rhs) noexcept
{
    return {
        lhs.x - rhs,
        lhs.y - rhs
    };
}

/**
 * The following macro is intended to allow the same syntax for constructing compound types
 * in both OpenCL and CUDA. In CUDA, we would write float2 { foo, bar }; but in OpenCL we
 * would write that (float2) { foo, bar };
 */
#define make_compound(_compound_type) _compound_type

/*

The ternary selection operator (?:) operates on three expressions (exp1 ? exp2 : exp3).
This operator evaluates the first expression exp1, which can be a scalar or vector result except float.
If all three expressions are scalar values, the C99 rules for ternary operator are followed. If the
result is a vector value, then this is equivalent to calling select(exp3, exp2, exp1). The select
function is described in Scalar and Vector Relational Functions. The second and third expressions
can be any type, as long their types match, or there is an implicit conversion that can be
applied to one of the expressions to make their types match, or one is a vector and the
other is a scalar and the scalar may be subject to the usual arithmetic conversion to the element
type used by the vector operand and widened to the same type as the vector type. This resulting
matching type is the type of the entire expression.
*/

#endif // __OPENCL_VERSION__
#endif // PORT_FROM_OPENCL_CUH_
