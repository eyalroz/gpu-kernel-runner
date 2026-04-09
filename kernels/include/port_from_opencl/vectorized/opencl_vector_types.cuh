/**
 * @file
 *
 * @brief Definitions for using vector types from originally-OpenCL-C code in 
 * CUDA code. This is part of a set of files which, included together, 
 * allow for using slightly-tweaked OpenCL C kernel code as CUDA code. 
 *
 * @copyright (c) 2020-2026, GE HealthCare
 * @copyright (c) 2020-2026, Eyal Rozenberg
 *
 * @license BSD 3-clause license; see the `LICENSE` file or
 * @url https://opensource.org/licenses/BSD-3-Clause
 *
 * @note Covered OpenCL C language spec section: 6.3.2. (Built-in Vector Data Types),
 * but without any builtin functions which take or return scalar types defined here.
 *
 * @todo Need to implement/add:
 *
 * - Support for vector sizes of 8 and 16
 * - More general vload's and vstore's than what we have now
 * - Reservation of vector types for lengths other than 2,3,4,8,16
 * - Other reservations as per 6.3.4. Reserved Data Types
 * - Enforcement (?) of 4-element alignment for 3-element vector types (as per §6.3.5)
 * - Named constructor idioms to replace vector type literals (as per S. §6.3.6)
 * - numeric subscripting - would require wrapping or subclassing the raw CUDA types
 *
 * @note No support for:
 * - .r/g/b/a member names (use .x/y/z/w)
 * - addressing multiple elements at once (e.g. `my_vec.zw = (float2) ( 1.2, 3.4 )`)
 */
#ifndef PORT_FROM_OPENCL_VECTOR_TYPES_CUH_
#define PORT_FROM_OPENCL_VECTOR_TYPES_CUH_

#include "opencl_scalar_types.cuh"

#ifndef __OPENCL_VERSION__

#ifndef make_compound
/**
 * The following macro is intended to allow the same syntax for constructing compound types
 * in both OpenCL and CUDA:
 *
 *   - In CUDA, we construct like so:   float2  { foo, bar };
 *   - in OpenCL we construct like so: (float2) { foo, bar };
 *
 * To bridge the difference, we always use a constructing macro, which is translated to
 * the appropriate syntax.
 */
#define make_compound(_compound_type) _compound_type
#endif

#ifdef PORT_FROM_OPENCL_ENABLE_HALF_PRECISION
// CUDA 13.0 and earlier (and perhaps also later), only offers headers
// which define half and half2 - but not half4
#include "../half4.cuh"
#endif

#if !defined(__CDT_PARSER__) && !defined (__JETBRAINS_IDE__)
// Parsers may fail to recognize a reasonable default-C++-headers path for kernel files
#include <cstdint>
#include <cstddef> // for size_t
#include <climits>
    // We don't really need these files directly, but failure to include them makes NVRTC
    // take another file rather than our NVRTC-safe climits stub
#endif

// CUDA offers its own vector types - whose names mostly overlap the OpenCL vector types; but
// they don't have quite the same functionality.
#include <vector_types.h>


template <typename I, I Value>
struct integral_constant {
    static constexpr I value = Value;
    using value_type = I;
    using type = integral_constant;

    constexpr operator value_type() const noexcept { return value; }
    constexpr value_type operator()() const noexcept { return value; }
};

namespace detail_ {
template <int Value>
using int_constant = integral_constant<int, Value>;
} // namespace detail_

using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

template <typename T> struct vectorizable;
template <> struct vectorizable<char>   : true_type {};
template <> struct vectorizable<uchar>  : true_type {};
template <> struct vectorizable<short>  : true_type {};
template <> struct vectorizable<ushort> : true_type {};
template <> struct vectorizable<int>    : true_type {};
template <> struct vectorizable<uint>   : true_type {};
template <> struct vectorizable<long>   : true_type {};
template <> struct vectorizable<ulong>  : true_type {};
template <typename T> struct vectorizable : false_type {};


// Vector type conversion functions

template <typename I > constexpr __device__ inline ushort convert_char    (I  v) noexcept { return (char)   v; }
template <typename I > constexpr __device__ inline ushort convert_uchar   (I  v) noexcept { return (uchar)  v; }
template <typename I > constexpr __device__ inline short  convert_short   (I  v) noexcept { return (short)  v; }
template <typename I > constexpr __device__ inline ushort convert_ushort  (I  v) noexcept { return (ushort) v; }
template <typename I > constexpr __device__ inline int    convert_int     (I  v) noexcept { return (int)    v; }
template <typename I > constexpr __device__ inline uint   convert_uint    (I  v) noexcept { return (uint)   v; }
template <typename I > constexpr __device__ inline float  convert_float   (I  v) noexcept { return (float)  v; }

template <typename I2> constexpr __device__ inline ushort2 convert_char2  (I2 v) noexcept { return { (char)   v.x, (char)   v.y }; }
template <typename I2> constexpr __device__ inline ushort2 convert_uchar2 (I2 v) noexcept { return { (uchar)  v.x, (uchar)  v.y }; }
template <typename I2> constexpr __device__ inline short2  convert_short2 (I2 v) noexcept { return { (short)  v.x, (short)  v.y }; }
template <typename I2> constexpr __device__ inline ushort2 convert_ushort2(I2 v) noexcept { return { (ushort) v.x, (ushort) v.y }; }
template <typename I2> constexpr __device__ inline int2    convert_int2   (I2 v) noexcept { return { (int)    v.x, (int)    v.y }; }
template <typename I2> constexpr __device__ inline uint2   convert_uint2  (I2 v) noexcept { return { (uint)   v.x, (uint)   v.y }; }
template <typename I2> constexpr __device__ inline float2  convert_float2 (I2 v) noexcept { return { (float)  v.x, (float)  v.y }; }

template <typename I3> constexpr __device__ inline float3  convert_float3 (I3 v) noexcept { return { (float)  v.x, (float)  v.y, (float)  v.z}; }
template <typename I3> constexpr __device__ inline int3    convert_int3   (I3 v) noexcept { return { (int)    v.x, (int)    v.y, (int)    v.z }; }
template <typename I3> constexpr __device__ inline uint3   convert_uint3  (I3 v) noexcept { return { (uint)   v.x, (uint)   v.y, (uint)   v.z }; }
template <typename I3> constexpr __device__ inline short3  convert_short3 (I3 v) noexcept { return { (short)  v.x, (short)  v.y, (short)  v.z }; }
template <typename I3> constexpr __device__ inline ushort3 convert_ushort3(I3 v) noexcept { return { (ushort) v.x, (ushort) v.y, (ushort) v.z }; }
template <typename I3> constexpr __device__ inline char3   convert_char3  (I3 v) noexcept { return { (char)   v.x, (char)   v.y, (char)   v.z }; }
template <typename I3> constexpr __device__ inline uchar3  convert_uchar3 (I3 v) noexcept { return { (uchar)  v.x, (uchar)  v.y, (uchar)  v.z }; }

template <typename I4> constexpr __device__ inline float4  convert_float4 (I4 v) noexcept { return { (float)  v.x, (float)  v.y, (float)  v.z, (float)   v.w }; }
template <typename I4> constexpr __device__ inline int4    convert_int4   (I4 v) noexcept { return { (int)    v.x, (int)    v.y, (int)    v.z, (int)     v.w }; }
template <typename I4> constexpr __device__ inline uint4   convert_uint4  (I4 v) noexcept { return { (uint)   v.x, (uint)   v.y, (uint)   v.z, (uint)    v.w }; }
template <typename I4> constexpr __device__ inline short4  convert_short4 (I4 v) noexcept { return { (short)  v.x, (short)  v.y, (short)  v.z, (short)   v.w }; }
template <typename I4> constexpr __device__ inline ushort4 convert_ushort4(I4 v) noexcept { return { (ushort) v.x, (ushort) v.y, (ushort) v.z, (ushort)  v.w }; }
template <typename I4> constexpr __device__ inline char4   convert_char4  (I4 v) noexcept { return { (char)   v.x, (char)   v.y, (char)   v.z, (char)    v.w }; }
template <typename I4> constexpr __device__ inline uchar4  convert_uchar4 (I4 v) noexcept { return { (uchar)  v.x, (uchar)  v.y, (uchar)  v.z, (uchar)   v.w }; }

// TODO: More scalar and vectorized type convert functions; see:
// https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/convert_T.html

constexpr __device__ inline float2 floor(const float2& v) { return { floorf(v.x), floorf(v.y) }; }
constexpr __device__ inline float4 floor(const float4& v)
{
    return { floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w) };
}

__device__ inline float  native_recip(float  x) { return __frcp_rn(x); }
__device__ inline double native_recip(double x) { return __drcp_rn(x); }

__device__ inline float  native_sqrt(float x)   { return sqrtf(x); }
__device__ inline double native_sqrt(double x)  { return sqrt(x);  }

__device__ inline float  native_rsqrt(float x)  { return rsqrtf(x); }
__device__ inline double native_rsqrt(double x) { return rsqrt(x);  }

//template <typename T, typename Selector>
//T select(T on_false, T on_true, Selector selector);


// TODO: Consider limiting the default implementation of vector_width to
// only support those types which can be vectorized/
template <typename T> struct opencl_vector_width;

template <> struct opencl_vector_width<char>   { enum { value = 1 }; };
template <> struct opencl_vector_width<uchar>  { enum { value = 1 }; };
template <> struct opencl_vector_width<short>  { enum { value = 1 }; };
template <> struct opencl_vector_width<ushort> { enum { value = 1 }; };
template <> struct opencl_vector_width<int>    { enum { value = 1 }; };
template <> struct opencl_vector_width<uint>   { enum { value = 1 }; };
template <> struct opencl_vector_width<long>   { enum { value = 1 }; };
template <> struct opencl_vector_width<ulong>  { enum { value = 1 }; };

template <> struct opencl_vector_width<char2>   { enum { value = 2 }; };
template <> struct opencl_vector_width<uchar2>  { enum { value = 2 }; };
template <> struct opencl_vector_width<short2>  { enum { value = 2 }; };
template <> struct opencl_vector_width<ushort2> { enum { value = 2 }; };
template <> struct opencl_vector_width<int2>    { enum { value = 2 }; };
template <> struct opencl_vector_width<uint2>   { enum { value = 2 }; };
template <> struct opencl_vector_width<long2>   { enum { value = 2 }; };
template <> struct opencl_vector_width<ulong2>  { enum { value = 2 }; };
#ifdef PORT_FROM_OPENCL_ENABLE_HALF_PRECISION
template <> struct opencl_vector_width<half2>   { enum { value = 2 }; };
#endif
template <> struct opencl_vector_width<float2>  { enum { value = 2 }; };
template <> struct opencl_vector_width<double2> { enum { value = 2 }; };

template <> struct opencl_vector_width<char3>   { enum { value = 3 }; };
template <> struct opencl_vector_width<uchar3>  { enum { value = 3 }; };
template <> struct opencl_vector_width<short3>  { enum { value = 3 }; };
template <> struct opencl_vector_width<ushort3> { enum { value = 3 }; };
template <> struct opencl_vector_width<int3>    { enum { value = 3 }; };
template <> struct opencl_vector_width<uint3>   { enum { value = 3 }; };
template <> struct opencl_vector_width<long3>   { enum { value = 3 }; };
template <> struct opencl_vector_width<ulong3>  { enum { value = 3 }; };
// No half3, it seems
//#ifdef PORT_FROM_OPENCL_ENABLE_HALF_PRECISION
//template <> struct opencl_vector_width<half3>   { enum { value = 3 }; };
//#endif
template <> struct opencl_vector_width<float3>  { enum { value = 3 }; };
template <> struct opencl_vector_width<double3> { enum { value = 3 }; };

template <> struct opencl_vector_width<char4>   { enum { value = 4 }; };
template <> struct opencl_vector_width<uchar4>  { enum { value = 4 }; };
template <> struct opencl_vector_width<short4>  { enum { value = 4 }; };
template <> struct opencl_vector_width<ushort4> { enum { value = 4 }; };
template <> struct opencl_vector_width<int4>    { enum { value = 4 }; };
template <> struct opencl_vector_width<uint4>   { enum { value = 4 }; };
template <> struct opencl_vector_width<long4>   { enum { value = 4 }; };
template <> struct opencl_vector_width<ulong4>  { enum { value = 4 }; };
#ifdef PORT_FROM_OPENCL_ENABLE_HALF_PRECISION
template <> struct opencl_vector_width<half4>   { enum { value = 4 }; };
#endif
template <> struct opencl_vector_width<float4>  { enum { value = 4 }; };
template <> struct opencl_vector_width<double4> { enum { value = 4 }; };

template <typename T> struct opencl_vector_width { };


template <size_t VectorWidth>
struct opencl_vectorized;

template <> struct opencl_vectorized<1>
{
    using short_ = short;
    using int_ = int;
    using long_ = long;
    using ushort_ = ushort;
    using uint = uint;
    using ulong_ = ulong;
#ifdef PORT_FROM_OPENCL_ENABLE_HALF_PRECISION
    using half_ = half;
#endif
    using float_ = float;
    using double_ = double;
};

template <> struct opencl_vectorized<2>
{
    using short_ = short2;
    using int_ = int2;
    using long_ = long2;
    using ushort_ = ushort2;
    using uint = uint2;
    using ulong_ = ulong2;
#ifdef PORT_FROM_OPENCL_ENABLE_HALF_PRECISION
    using half_ = half2;
#endif
    using float_ = float2;
    using double_ = double2;
};

template <> struct opencl_vectorized<3>
{
    using short_ = short3;
    using int_ = int3;
    using long_ = long3;
    using ushort_ = ushort3;
    using uint = uint3;
    using ulong_ = ulong3;
// No half3, it seems
//#ifdef PORT_FROM_OPENCL_ENABLE_HALF_PRECISION
//    using half_ = half3;
//#endif
    using float_ = float3;
    using double_ = double3;
};

template <> struct opencl_vectorized<4>
{
    using short_ = short4;
    using int_ = int4;
    using long_ = long4;
    using ushort_ = ushort4;
    using uint = uint4;
    using ulong_ = ulong4;
#ifdef PORT_FROM_OPENCL_ENABLE_HALF_PRECISION
    using half_ = half4;
#endif
    using float_ = float4;
    using double_ = double4;
};

template <typename OpenCLVector>
using to_ints = typename opencl_vectorized<opencl_vector_width<OpenCLVector>::value>::int_;


template <typename Scalar>
constexpr __device__ inline Scalar select(
    Scalar on_false,
    Scalar on_true,
    int selector)
{
	static_assert(opencl_vector_width<Scalar>::value > 1, "Don't use this on vector types");
    return selector ? on_true : on_false;
}
//
//namespace detail_ {
//
//template <typename T> struct is_gentype_f;
//template <typename T> struct is_gentype_f;
//
//{
//};
//
//} // namespace detail_

namespace detail_ {

template <typename OpenCLVector, size_t VectorWidth>
constexpr inline typename opencl_vectorized<VectorWidth>::int_ isequal(OpenCLVector lhs, OpenCLVector rhs);

template <typename OpenCLVector> constexpr inline opencl_vectorized<1>::int_ isequal(int_constant<1>, OpenCLVector lhs, OpenCLVector rhs) { return lhs == rhs; }
template <typename OpenCLVector> constexpr inline opencl_vectorized<2>::int_ isequal(int_constant<2>, OpenCLVector lhs, OpenCLVector rhs) { return { lhs.x == rhs.x, lhs.y == rhs.y }; }
template <typename OpenCLVector> constexpr inline opencl_vectorized<3>::int_ isequal(int_constant<3>, OpenCLVector lhs, OpenCLVector rhs) { return { lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z }; }
template <typename OpenCLVector> constexpr inline opencl_vectorized<4>::int_ isequal(int_constant<4>, OpenCLVector lhs, OpenCLVector rhs) { return { lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z, lhs.w == rhs.w }; }

template <typename OpenCLVector, size_t VectorWidth>
constexpr inline typename opencl_vectorized<VectorWidth>::int_ isnotequal(OpenCLVector lhs, OpenCLVector rhs);

template <typename OpenCLVector> constexpr inline opencl_vectorized<1>::int_ isnotequal(int_constant<1>, OpenCLVector lhs, OpenCLVector rhs) { return lhs != rhs; }
template <typename OpenCLVector> constexpr inline opencl_vectorized<2>::int_ isnotequal(int_constant<2>, OpenCLVector lhs, OpenCLVector rhs) { return { lhs.x != rhs.x, lhs.y != rhs.y }; }
template <typename OpenCLVector> constexpr inline opencl_vectorized<3>::int_ isnotequal(int_constant<3>, OpenCLVector lhs, OpenCLVector rhs) { return { lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z }; }
template <typename OpenCLVector> constexpr inline opencl_vectorized<4>::int_ isnotequal(int_constant<4>, OpenCLVector lhs, OpenCLVector rhs) { return { lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z, lhs.w != rhs.w }; }

template <typename OpenCLVector, size_t VectorWidth>
constexpr inline typename opencl_vectorized<VectorWidth>::int_ isless(OpenCLVector lhs, OpenCLVector rhs);

template <typename OpenCLVector> constexpr inline opencl_vectorized<1>::int_ isless(int_constant<1>, OpenCLVector lhs, OpenCLVector rhs) { return lhs < rhs; }
template <typename OpenCLVector> constexpr inline opencl_vectorized<2>::int_ isless(int_constant<2>, OpenCLVector lhs, OpenCLVector rhs) { return {lhs.x < rhs.x, lhs.y < rhs.y}; }
template <typename OpenCLVector> constexpr inline opencl_vectorized<3>::int_ isless(int_constant<3>, OpenCLVector lhs, OpenCLVector rhs) { return {lhs.x < rhs.x, lhs.y < rhs.y, lhs.z < rhs.z}; }
template <typename OpenCLVector> constexpr inline opencl_vectorized<4>::int_ isless(int_constant<4>, OpenCLVector lhs, OpenCLVector rhs) { return { lhs.x < rhs.x, lhs.y < rhs.y, lhs.z < rhs.z, lhs.w < rhs.w }; }

template <typename OpenCLVector, size_t VectorWidth>
constexpr inline bool all(OpenCLVector v);

template <typename OpenCLVector> constexpr inline bool all(int_constant<1>, OpenCLVector v) { return v.x; }
template <typename OpenCLVector> constexpr inline bool all(int_constant<2>, OpenCLVector v) { return v.x and v.y; }
template <typename OpenCLVector> constexpr inline bool all(int_constant<3>, OpenCLVector v) { return v.x and v.y and v.z; }
template <typename OpenCLVector> constexpr inline bool all(int_constant<4>, OpenCLVector v) { return v.x and v.y and v.z and v.w; }

template <typename OpenCLVector, size_t VectorWidth>
constexpr inline bool any(OpenCLVector v);

template <typename OpenCLVector> constexpr inline bool any(int_constant<1>, OpenCLVector v) { return v.x; }
template <typename OpenCLVector> constexpr inline bool any(int_constant<2>, OpenCLVector v) { return v.x or v.y; }
template <typename OpenCLVector> constexpr inline bool any(int_constant<3>, OpenCLVector v) { return v.x or v.y or v.z; }
template <typename OpenCLVector> constexpr inline bool any(int_constant<4>, OpenCLVector v) { return v.x or v.y or v.z or v.w; }

} // namespace detail_

template <typename OpenCLVector>
constexpr inline to_ints<OpenCLVector> isequal(OpenCLVector x, OpenCLVector y)
{
    enum { vector_width = opencl_vector_width<OpenCLVector>::value };
    using vector_width_type = detail_::int_constant<vector_width>;
    return detail_::isequal(vector_width_type{}, x,y);
}

template <typename OpenCLVector>
constexpr inline to_ints<OpenCLVector> isnotequal(OpenCLVector x, OpenCLVector y)
{
    enum { vector_width = opencl_vector_width<OpenCLVector>::value };
    using vector_width_type = detail_::int_constant<vector_width>;
    return detail_::isnotequal(vector_width_type{}, x,y);
}

template <typename OpenCLVector>
constexpr inline to_ints<OpenCLVector> isless(OpenCLVector x, OpenCLVector y)
{
    enum { vector_width = opencl_vector_width<OpenCLVector>::value };
    using vector_width_type = detail_::int_constant<vector_width>;
    return detail_::isless(vector_width_type{}, x, y);
}

template <typename OpenCLVector>
constexpr inline bool all(OpenCLVector v)
{
    enum { vector_width = opencl_vector_width<OpenCLVector>::value };
    using vector_width_type = detail_::int_constant<vector_width>;
    return detail_::all(vector_width_type{}, v);
}

template <typename OpenCLVector>
constexpr inline bool any(OpenCLVector v)
{
    enum { vector_width = opencl_vector_width<OpenCLVector>::value };
    using vector_width_type = detail_::int_constant<vector_width>;
    return detail_::any(vector_width_type{}, v);
}


template <typename OpenCLVector>
constexpr inline to_ints<OpenCLVector> isgreater(OpenCLVector x, OpenCLVector y)
{
    return isless(y, x);
}

template <typename OpenCLVector>
constexpr inline to_ints<OpenCLVector> isgreaterqual(OpenCLVector x, OpenCLVector y)
{
    return not isless(x, y);
}

template <typename OpenCLVector>
constexpr inline to_ints<OpenCLVector> islessequal(OpenCLVector x, OpenCLVector y)
{
    return not isgreater(x, y);
}

// Note: These are semantically-unsound implementations, which do
// not assume actual floatn alignment, and may result in a subotimal
// choice of SASS instructions

constexpr __device__ inline float2 vload2(size_t offset, const float* p)
{
return { p[offset], p[offset+1] };
//    return reinterpret_cast<const float2*>(p)[offset];
}

constexpr __device__ inline void vstore2(const float2& value, size_t offset, float* p)
{
    p[offset  ] = value.x;
    p[offset+1] = value.y;
//    reinterpret_cast<float2*>(p)[offset] = value;
}

constexpr __device__ inline float3 vload3(size_t offset, const float* p)
{
return { p[offset], p[offset+1], p[offset+2] };
//    return reinterpret_cast<const float3*>(p)[offset];
}

constexpr __device__ inline void vstore3(const float3& value, size_t offset, float* p)
{
    p[offset  ] = value.x;
    p[offset+1] = value.y;
    p[offset+2] = value.z;
//    reinterpret_cast<float3*>(p)[offset] = value;
}

constexpr  __device__ inline float4 vload4(size_t offset, const float* p)
{
return { p[offset], p[offset+1], p[offset+2], p[offset+3] };
//    return reinterpret_cast<const float4*>(p)[offset];
}

constexpr __device__ inline void vstore4(const float4& value, size_t offset, float* p)
{
    p[offset  ] = value.x;
    p[offset+1] = value.y;
    p[offset+2] = value.z;
    p[offset+3] = value.w;
//    reinterpret_cast<float4*>(p)[offset] = value;
}

/* Missing vector-type functions: bitselect, any, all, signbit, isordered, isunordered, bitselect */


// Arithmetic, comparison and assignment operators for vectorized types

#ifndef VECTORIZED_TYPES_BASIC_OPERATORS
#define VECTORIZED_TYPES_BASIC_OPERATORS

// TODO: Add missing comparison operators; Define a bool
// Note: Not "undoing" integer promotions

// short2

// short2 with short2

constexpr __device__ inline int2 operator+(short2 lhs, short2 rhs) noexcept { return { lhs.x + rhs.x, lhs.y + rhs.y }; }
constexpr __device__ inline int2 operator-(short2 lhs, short2 rhs) noexcept { return { lhs.x - rhs.x, lhs.y - rhs.y }; }
constexpr __device__ inline int2 operator*(short2 lhs, short2 rhs) noexcept { return { lhs.x * rhs.x, lhs.y * rhs.y }; }
constexpr __device__ inline int2 operator/(short2 lhs, short2 rhs) noexcept { return { lhs.x / rhs.x, lhs.y / rhs.y }; }

constexpr __device__ inline int2 operator&(short2 lhs, short2 rhs) noexcept { return { lhs.x & rhs.x, lhs.y & rhs.y }; }
constexpr __device__ inline int2 operator|(short2 lhs, short2 rhs) noexcept { return { lhs.x | rhs.x, lhs.y | rhs.y }; }
constexpr __device__ inline int2 operator^(short2 lhs, short2 rhs) noexcept { return { lhs.x ^ rhs.x, lhs.y ^ rhs.y }; }
constexpr __device__ inline int2 operator~(short2 lhs) noexcept { return { ~lhs.x, ~lhs.y }; }

constexpr __device__ inline short2& operator+=(short2& lhs, short2 rhs) noexcept { lhs = convert_short2(lhs + rhs); return lhs; }
constexpr __device__ inline short2& operator-=(short2& lhs, short2 rhs) noexcept { lhs = convert_short2(lhs - rhs); return lhs; }

constexpr __device__ inline int2 operator< (short2 lhs, short2 rhs) noexcept { return { lhs.x <  rhs.x, lhs.y <  rhs.y }; }
constexpr __device__ inline int2 operator<=(short2 lhs, short2 rhs) noexcept { return { lhs.x <= rhs.x, lhs.y <= rhs.y }; }
constexpr __device__ inline int2 operator> (short2 lhs, short2 rhs) noexcept { return { lhs.x >  rhs.x, lhs.y >  rhs.y }; }
constexpr __device__ inline int2 operator>=(short2 lhs, short2 rhs) noexcept { return { lhs.x >= rhs.x, lhs.y >= rhs.y }; }
constexpr __device__ inline int2 operator==(short2 lhs, short2 rhs) noexcept { return { lhs.x == rhs.x, lhs.y == rhs.y }; }
constexpr __device__ inline int2 operator!=(short2 lhs, short2 rhs) noexcept { return { lhs.x != rhs.x, lhs.y != rhs.y }; }

// short with short2

constexpr __device__ inline int2 operator+(short lhs, short2 rhs) noexcept { return { lhs + rhs.x, lhs + rhs.y }; }
constexpr __device__ inline int2 operator-(short lhs, short2 rhs) noexcept { return { lhs - rhs.x, lhs - rhs.y }; }
constexpr __device__ inline int2 operator*(short lhs, short2 rhs) noexcept { return { lhs * rhs.x, lhs * rhs.y }; }
constexpr __device__ inline int2 operator/(short lhs, short2 rhs) noexcept { return { lhs / rhs.x, lhs / rhs.y }; }

constexpr __device__ inline int2 operator&(short lhs, short2 rhs) noexcept { return { lhs & rhs.x, lhs & rhs.y }; }
constexpr __device__ inline int2 operator|(short lhs, short2 rhs) noexcept { return { lhs | rhs.x, lhs | rhs.y }; }
constexpr __device__ inline int2 operator^(short lhs, short2 rhs) noexcept { return { lhs ^ rhs.x, lhs ^ rhs.y }; }

constexpr __device__ inline int2 operator< (short lhs, short2 rhs) noexcept { return { lhs <  rhs.x, lhs <  rhs.y }; }
constexpr __device__ inline int2 operator<=(short lhs, short2 rhs) noexcept { return { lhs <= rhs.x, lhs <= rhs.y }; }
constexpr __device__ inline int2 operator> (short lhs, short2 rhs) noexcept { return { lhs >  rhs.x, lhs >  rhs.y }; }
constexpr __device__ inline int2 operator>=(short lhs, short2 rhs) noexcept { return { lhs >= rhs.x, lhs >= rhs.y }; }
constexpr __device__ inline int2 operator==(short lhs, short2 rhs) noexcept { return { lhs == rhs.x, lhs == rhs.y }; }
constexpr __device__ inline int2 operator!=(short lhs, short2 rhs) noexcept { return { lhs != rhs.x, lhs != rhs.y }; }

// short2 with short

constexpr __device__ inline int2 operator+(short2 lhs, short rhs) noexcept { return { lhs.x +  rhs, lhs.y + rhs }; }
constexpr __device__ inline int2 operator-(short2 lhs, short rhs) noexcept { return { lhs.x -  rhs, lhs.y - rhs }; }
constexpr __device__ inline int2 operator*(short2 lhs, short rhs) noexcept { return { lhs.x *  rhs, lhs.y * rhs }; }
constexpr __device__ inline int2 operator/(short2 lhs, short rhs) noexcept { return { lhs.x /  rhs, lhs.y / rhs }; }

constexpr __device__ inline int2 operator&(short2 lhs, short rhs) noexcept { return { lhs.x & rhs, lhs.y & rhs }; }
constexpr __device__ inline int2 operator|(short2 lhs, short rhs) noexcept { return { lhs.x | rhs, lhs.y | rhs }; }
constexpr __device__ inline int2 operator^(short2 lhs, short rhs) noexcept { return { lhs.x ^ rhs, lhs.y ^ rhs }; }

constexpr __device__ inline short2& operator+=(short2& lhs, short rhs) noexcept { lhs = convert_short2(lhs + rhs); return lhs; }
constexpr __device__ inline short2& operator-=(short2& lhs, short rhs) noexcept { lhs = convert_short2(lhs - rhs); return lhs; }

constexpr __device__ inline int2 operator< (short2 lhs, short rhs) noexcept { return { lhs.x <   rhs, lhs.y <  rhs }; }
constexpr __device__ inline int2 operator<=(short2 lhs, short rhs) noexcept { return { lhs.x <=  rhs, lhs.y <= rhs }; }
constexpr __device__ inline int2 operator> (short2 lhs, short rhs) noexcept { return { lhs.x >   rhs, lhs.y >  rhs }; }
constexpr __device__ inline int2 operator>=(short2 lhs, short rhs) noexcept { return { lhs.x >=  rhs, lhs.y >= rhs }; }
constexpr __device__ inline int2 operator==(short2 lhs, short rhs) noexcept { return { lhs.x ==  rhs, lhs.y == rhs }; }
constexpr __device__ inline int2 operator!=(short2 lhs, short rhs) noexcept { return { lhs.x !=  rhs, lhs.y != rhs }; }

// ushort2

// ushort2 with ushort2

constexpr __device__ inline int2 operator+(ushort2 lhs, ushort2 rhs) noexcept { return { lhs.x + rhs.x, lhs.y + rhs.y }; }
constexpr __device__ inline int2 operator-(ushort2 lhs, ushort2 rhs) noexcept { return { lhs.x - rhs.x, lhs.y - rhs.y }; }
constexpr __device__ inline int2 operator/(ushort2 lhs, ushort2 rhs) noexcept { return { lhs.x / rhs.x, lhs.y / rhs.y }; }

constexpr __device__ inline int2 operator&(ushort2 lhs, ushort2 rhs) noexcept { return { lhs.x & rhs.x, lhs.y & rhs.y }; }
constexpr __device__ inline int2 operator|(ushort2 lhs, ushort2 rhs) noexcept { return { lhs.x | rhs.x, lhs.y | rhs.y }; }
constexpr __device__ inline int2 operator^(ushort2 lhs, ushort2 rhs) noexcept { return { lhs.x ^ rhs.x, lhs.y ^ rhs.y }; }
constexpr __device__ inline int2 operator~(ushort2 lhs) noexcept { return { ~lhs.x, ~lhs.y }; }

constexpr __device__ inline ushort2& operator+=(ushort2& lhs, ushort2 rhs) noexcept { lhs = convert_ushort2(lhs + rhs); return lhs; }
constexpr __device__ inline ushort2& operator-=(ushort2& lhs, ushort2 rhs) noexcept { lhs = convert_ushort2(lhs - rhs); return lhs; }

constexpr __device__ inline int2 operator< (ushort2 lhs, ushort2 rhs) noexcept { return { lhs.x <  rhs.x, lhs.y <  rhs.y }; }
constexpr __device__ inline int2 operator<=(ushort2 lhs, ushort2 rhs) noexcept { return { lhs.x <= rhs.x, lhs.y <= rhs.y }; }
constexpr __device__ inline int2 operator> (ushort2 lhs, ushort2 rhs) noexcept { return { lhs.x >  rhs.x, lhs.y >  rhs.y }; }
constexpr __device__ inline int2 operator>=(ushort2 lhs, ushort2 rhs) noexcept { return { lhs.x >= rhs.x, lhs.y >= rhs.y }; }
constexpr __device__ inline int2 operator==(ushort2 lhs, ushort2 rhs) noexcept { return { lhs.x == rhs.x, lhs.y == rhs.y }; }
constexpr __device__ inline int2 operator!=(ushort2 lhs, ushort2 rhs) noexcept { return { lhs.x != rhs.x, lhs.y != rhs.y }; }

// ushort with ushort2

constexpr __device__ inline int2 operator+(ushort lhs, ushort2 rhs) noexcept { return { lhs + rhs.x, lhs + rhs.y }; }
constexpr __device__ inline int2 operator-(ushort lhs, ushort2 rhs) noexcept { return { lhs - rhs.x, lhs - rhs.y }; }
constexpr __device__ inline int2 operator*(ushort lhs, ushort2 rhs) noexcept { return { lhs * rhs.x, lhs * rhs.y }; }
constexpr __device__ inline int2 operator/(ushort lhs, ushort2 rhs) noexcept { return { lhs / rhs.x, lhs / rhs.y }; }

constexpr __device__ inline int2 operator&(ushort lhs, ushort2 rhs) noexcept { return { lhs & rhs.x, lhs & rhs.y }; }
constexpr __device__ inline int2 operator|(ushort lhs, ushort2 rhs) noexcept { return { lhs | rhs.x, lhs | rhs.y }; }
constexpr __device__ inline int2 operator^(ushort lhs, ushort2 rhs) noexcept { return { lhs ^ rhs.x, lhs ^ rhs.y }; }

constexpr __device__ inline int2 operator< (ushort lhs, ushort2 rhs) noexcept { return { lhs <  rhs.x, lhs <  rhs.y }; }
constexpr __device__ inline int2 operator<=(ushort lhs, ushort2 rhs) noexcept { return { lhs <= rhs.x, lhs <= rhs.y }; }
constexpr __device__ inline int2 operator> (ushort lhs, ushort2 rhs) noexcept { return { lhs >  rhs.x, lhs >  rhs.y }; }
constexpr __device__ inline int2 operator>=(ushort lhs, ushort2 rhs) noexcept { return { lhs >= rhs.x, lhs >= rhs.y }; }
constexpr __device__ inline int2 operator==(ushort lhs, ushort2 rhs) noexcept { return { lhs == rhs.x, lhs == rhs.y }; }
constexpr __device__ inline int2 operator!=(ushort lhs, ushort2 rhs) noexcept { return { lhs != rhs.x, lhs != rhs.y }; }

// ushort2 with ushort

constexpr __device__ inline int2 operator+(ushort2 lhs, ushort rhs) noexcept { return { lhs.x + rhs, lhs.y + rhs }; }
constexpr __device__ inline int2 operator-(ushort2 lhs, ushort rhs) noexcept { return { lhs.x - rhs, lhs.y - rhs }; }
constexpr __device__ inline int2 operator*(ushort2 lhs, ushort rhs) noexcept { return { lhs.x * rhs, lhs.y * rhs }; }
constexpr __device__ inline int2 operator/(ushort2 lhs, ushort rhs) noexcept { return { lhs.x / rhs, lhs.y / rhs }; }

constexpr __device__ inline int2 operator&(ushort2 lhs, ushort rhs) noexcept { return { lhs.x & rhs, lhs.y & rhs }; }
constexpr __device__ inline int2 operator|(ushort2 lhs, ushort rhs) noexcept { return { lhs.x | rhs, lhs.y | rhs }; }
constexpr __device__ inline int2 operator^(ushort2 lhs, ushort rhs) noexcept { return { lhs.x ^ rhs, lhs.y ^ rhs }; }

constexpr __device__ inline ushort2& operator+=(ushort2& lhs, ushort rhs) noexcept { lhs = convert_ushort2(lhs + rhs); return lhs; }
constexpr __device__ inline ushort2& operator-=(ushort2& lhs, ushort rhs) noexcept { lhs = convert_ushort2(lhs - rhs); return lhs; }

constexpr __device__ inline int2 operator< (ushort2 lhs, ushort rhs) noexcept { return { lhs.x <  rhs, lhs.y <  rhs }; }
constexpr __device__ inline int2 operator<=(ushort2 lhs, ushort rhs) noexcept { return { lhs.x <= rhs, lhs.y <= rhs }; }
constexpr __device__ inline int2 operator> (ushort2 lhs, ushort rhs) noexcept { return { lhs.x >  rhs, lhs.y >  rhs }; }
constexpr __device__ inline int2 operator>=(ushort2 lhs, ushort rhs) noexcept { return { lhs.x >= rhs, lhs.y >= rhs }; }
constexpr __device__ inline int2 operator==(ushort2 lhs, ushort rhs) noexcept { return { lhs.x == rhs, lhs.y == rhs }; }
constexpr __device__ inline int2 operator!=(ushort2 lhs, ushort rhs) noexcept { return { lhs.x != rhs, lhs.y != rhs }; }

// int2

// int2 with int2

constexpr __device__ inline int2 operator+(int2 lhs, int2 rhs) noexcept { return { lhs.x + rhs.x, lhs.y + rhs.y }; }
constexpr __device__ inline int2 operator-(int2 lhs, int2 rhs) noexcept { return { lhs.x - rhs.x, lhs.y - rhs.y }; }
constexpr __device__ inline int2 operator*(int2 lhs, int2 rhs) noexcept { return { lhs.x * rhs.x, lhs.y * rhs.y }; }
constexpr __device__ inline int2 operator/(int2 lhs, int2 rhs) noexcept { return { lhs.x / rhs.x, lhs.y / rhs.y }; }

constexpr __device__ inline int2 operator&(int2 lhs, int2 rhs) noexcept { return { lhs.x & rhs.x, lhs.y & rhs.y }; }
constexpr __device__ inline int2 operator|(int2 lhs, int2 rhs) noexcept { return { lhs.x | rhs.x, lhs.y | rhs.y }; }
constexpr __device__ inline int2 operator^(int2 lhs, int2 rhs) noexcept { return { lhs.x ^ rhs.x, lhs.y ^ rhs.y }; }
constexpr __device__ inline int2 operator~(int2 lhs) noexcept { return { ~lhs.x, ~lhs.y }; }

constexpr __device__ inline int2& operator+=(int2& lhs, int2 rhs) noexcept { lhs = convert_int2(lhs + rhs); return lhs; }
constexpr __device__ inline int2& operator-=(int2& lhs, int2 rhs) noexcept { lhs = convert_int2(lhs - rhs); return lhs; }

constexpr __device__ inline int2 operator< (int2 lhs, int2 rhs) noexcept { return { lhs.x <  rhs.x, lhs.y <  rhs.y }; }
constexpr __device__ inline int2 operator<=(int2 lhs, int2 rhs) noexcept { return { lhs.x <= rhs.x, lhs.y <= rhs.y }; }
constexpr __device__ inline int2 operator> (int2 lhs, int2 rhs) noexcept { return { lhs.x >  rhs.x, lhs.y >  rhs.y }; }
constexpr __device__ inline int2 operator>=(int2 lhs, int2 rhs) noexcept { return { lhs.x >= rhs.x, lhs.y >= rhs.y }; }
constexpr __device__ inline int2 operator==(int2 lhs, int2 rhs) noexcept { return { lhs.x == rhs.x, lhs.y == rhs.y }; }
constexpr __device__ inline int2 operator!=(int2 lhs, int2 rhs) noexcept { return { lhs.x != rhs.x, lhs.y != rhs.y }; }

// int with int2

constexpr __device__ inline int2 operator+(int lhs, int2 rhs) noexcept { return { lhs + rhs.x, lhs + rhs.y }; }
constexpr __device__ inline int2 operator-(int lhs, int2 rhs) noexcept { return { lhs - rhs.x, lhs - rhs.y }; }
constexpr __device__ inline int2 operator*(int lhs, int2 rhs) noexcept { return { lhs * rhs.x, lhs * rhs.y }; }
constexpr __device__ inline int2 operator/(int lhs, int2 rhs) noexcept { return { lhs / rhs.x, lhs / rhs.y }; }

constexpr __device__ inline int2 operator&(int lhs, int2 rhs) noexcept { return { lhs & rhs.x,lhs & rhs.y }; }
constexpr __device__ inline int2 operator|(int lhs, int2 rhs) noexcept { return { lhs | rhs.x, lhs | rhs.y }; }
constexpr __device__ inline int2 operator^(int lhs, int2 rhs) noexcept { return { lhs ^ rhs.x, lhs ^ rhs.y }; }

constexpr __device__ inline int2 operator< (int lhs, int2 rhs) noexcept { return { lhs <  rhs.x, lhs <  rhs.y }; }
constexpr __device__ inline int2 operator<=(int lhs, int2 rhs) noexcept { return { lhs <= rhs.x, lhs <= rhs.y }; }
constexpr __device__ inline int2 operator> (int lhs, int2 rhs) noexcept { return { lhs >  rhs.x, lhs >  rhs.y }; }
constexpr __device__ inline int2 operator>=(int lhs, int2 rhs) noexcept { return { lhs >= rhs.x, lhs >= rhs.y }; }
constexpr __device__ inline int2 operator==(int lhs, int2 rhs) noexcept { return { lhs == rhs.x, lhs == rhs.y }; }
constexpr __device__ inline int2 operator!=(int lhs, int2 rhs) noexcept { return { lhs != rhs.x, lhs != rhs.y }; }

// int2 with int

constexpr __device__ inline int2 operator+(int2 lhs, int rhs) noexcept { return { lhs.x + rhs, lhs.y + rhs }; }
constexpr __device__ inline int2 operator-(int2 lhs, int rhs) noexcept { return { lhs.x - rhs, lhs.y - rhs }; }
constexpr __device__ inline int2 operator*(int2 lhs, int rhs) noexcept { return { lhs.x * rhs, lhs.y * rhs }; }
constexpr __device__ inline int2 operator/(int2 lhs, int rhs) noexcept { return { lhs.x / rhs, lhs.y / rhs }; }

constexpr __device__ inline int2 operator&(int2 lhs, int rhs) noexcept { return { lhs.x & rhs, lhs.y & rhs }; }
constexpr __device__ inline int2 operator|(int2 lhs, int rhs) noexcept { return { lhs.x | rhs, lhs.y | rhs }; }
constexpr __device__ inline int2 operator^(int2 lhs, int rhs) noexcept { return { lhs.x ^ rhs, lhs.y ^ rhs }; }

constexpr __device__ inline int2& operator+=(int2& lhs, int rhs) noexcept { lhs = convert_int2(lhs + rhs); return lhs; }
constexpr __device__ inline int2& operator-=(int2& lhs, int rhs) noexcept { lhs = convert_int2(lhs - rhs); return lhs; }

constexpr __device__ inline int2 operator< (int2 lhs, int rhs) noexcept { return { lhs.x <  rhs, lhs.y <  rhs }; }
constexpr __device__ inline int2 operator<=(int2 lhs, int rhs) noexcept { return { lhs.x <= rhs, lhs.y <= rhs }; }
constexpr __device__ inline int2 operator> (int2 lhs, int rhs) noexcept { return { lhs.x >  rhs, lhs.y >  rhs }; }
constexpr __device__ inline int2 operator>=(int2 lhs, int rhs) noexcept { return { lhs.x >= rhs, lhs.y >= rhs }; }
constexpr __device__ inline int2 operator==(int2 lhs, int rhs) noexcept { return { lhs.x == rhs, lhs.y == rhs }; }
constexpr __device__ inline int2 operator!=(int2 lhs, int rhs) noexcept { return { lhs.x != rhs, lhs.y != rhs }; }

// uint2

// uint2 with uint2

constexpr __device__ inline uint2 operator+(uint2 lhs, uint2 rhs) noexcept { return { lhs.x + rhs.x, lhs.y + rhs.y }; }
constexpr __device__ inline uint2 operator-(uint2 lhs, uint2 rhs) noexcept { return { lhs.x - rhs.x, lhs.y - rhs.y }; }
constexpr __device__ inline uint2 operator*(uint2 lhs, uint2 rhs) noexcept { return { lhs.x * rhs.x, lhs.y * rhs.y }; }
constexpr __device__ inline uint2 operator/(uint2 lhs, uint2 rhs) noexcept { return { lhs.x / rhs.x, lhs.y / rhs.y }; }

constexpr __device__ inline uint2 operator&(uint2 lhs, uint2 rhs) noexcept { return { lhs.x & rhs.x, lhs.y & rhs.y }; }
constexpr __device__ inline uint2 operator|(uint2 lhs, uint2 rhs) noexcept { return { lhs.x | rhs.x, lhs.y | rhs.y }; }
constexpr __device__ inline uint2 operator^(uint2 lhs, uint2 rhs) noexcept { return { lhs.x ^ rhs.x, lhs.y ^ rhs.y }; }
constexpr __device__ inline uint2 operator~(uint2 lhs) noexcept { return { (uint) ~lhs.x, (uint) ~lhs.y }; }

constexpr __device__ inline uint2& operator+=(uint2& lhs, uint2 rhs) noexcept { lhs = convert_uint2(lhs + rhs); return lhs; }
constexpr __device__ inline uint2& operator-=(uint2& lhs, uint2 rhs) noexcept { lhs = convert_uint2(lhs - rhs); return lhs; }

constexpr __device__ inline int2 operator< (uint2 lhs, uint2 rhs) noexcept { return { lhs.x <  rhs.x, lhs.y <  rhs.y }; }
constexpr __device__ inline int2 operator<=(uint2 lhs, uint2 rhs) noexcept { return { lhs.x <= rhs.x, lhs.y <= rhs.y }; }
constexpr __device__ inline int2 operator> (uint2 lhs, uint2 rhs) noexcept { return { lhs.x >  rhs.x, lhs.y >  rhs.y }; }
constexpr __device__ inline int2 operator>=(uint2 lhs, uint2 rhs) noexcept { return { lhs.x >= rhs.x, lhs.y >= rhs.y }; }
constexpr __device__ inline int2 operator==(uint2 lhs, uint2 rhs) noexcept { return { lhs.x == rhs.x, lhs.y == rhs.y }; }
constexpr __device__ inline int2 operator!=(uint2 lhs, uint2 rhs) noexcept { return { lhs.x != rhs.x, lhs.y != rhs.y }; }

// uint with uint2

constexpr __device__ inline uint2 operator+(uint lhs, uint2 rhs) noexcept { return { lhs + rhs.x, lhs + rhs.y }; }
constexpr __device__ inline uint2 operator-(uint lhs, uint2 rhs) noexcept { return { lhs - rhs.x, lhs - rhs.y }; }
constexpr __device__ inline uint2 operator*(uint lhs, uint2 rhs) noexcept { return { lhs * rhs.x, lhs * rhs.y }; }
constexpr __device__ inline uint2 operator/(uint lhs, uint2 rhs) noexcept { return { lhs / rhs.x, lhs / rhs.y }; }

constexpr __device__ inline uint2 operator&(uint lhs, uint2 rhs) noexcept { return { lhs & rhs.x, lhs & rhs.y }; }
constexpr __device__ inline uint2 operator|(uint lhs, uint2 rhs) noexcept { return { lhs | rhs.x, lhs | rhs.y }; }
constexpr __device__ inline uint2 operator^(uint lhs, uint2 rhs) noexcept { return { lhs ^ rhs.x, lhs ^ rhs.y }; }

constexpr __device__ inline int2 operator< (uint lhs, uint2 rhs) noexcept { return { lhs <  rhs.x, lhs <  rhs.y }; }
constexpr __device__ inline int2 operator<=(uint lhs, uint2 rhs) noexcept { return { lhs <= rhs.x, lhs <= rhs.y }; }
constexpr __device__ inline int2 operator> (uint lhs, uint2 rhs) noexcept { return { lhs >  rhs.x, lhs >  rhs.y }; }
constexpr __device__ inline int2 operator>=(uint lhs, uint2 rhs) noexcept { return { lhs >= rhs.x, lhs >= rhs.y }; }
constexpr __device__ inline int2 operator==(uint lhs, uint2 rhs) noexcept { return { lhs == rhs.x, lhs == rhs.y }; }
constexpr __device__ inline int2 operator!=(uint lhs, uint2 rhs) noexcept { return { lhs != rhs.x, lhs != rhs.y }; }

// uint2 with uint

constexpr __device__ inline uint2 operator+(uint2 lhs, uint rhs) noexcept { return { lhs.x + rhs, lhs.y + rhs }; }
constexpr __device__ inline uint2 operator-(uint2 lhs, uint rhs) noexcept { return { lhs.x - rhs, lhs.y - rhs }; }
constexpr __device__ inline uint2 operator*(uint2 lhs, uint rhs) noexcept { return { lhs.x * rhs, lhs.y * rhs }; }
constexpr __device__ inline uint2 operator/(uint2 lhs, uint rhs) noexcept { return { lhs.x / rhs, lhs.y / rhs }; }

constexpr __device__ inline uint2 operator&(uint2 lhs, uint rhs) noexcept { return { lhs.x & rhs, lhs.y & rhs }; }
constexpr __device__ inline uint2 operator|(uint2 lhs, uint rhs) noexcept { return { lhs.x | rhs, lhs.y | rhs }; }
constexpr __device__ inline uint2 operator^(uint2 lhs, uint rhs) noexcept { return { lhs.x ^ rhs, lhs.y ^ rhs }; }

constexpr __device__ inline uint2& operator+=(uint2& lhs, uint rhs) noexcept { lhs = convert_uint2(lhs + rhs); return lhs; }
constexpr __device__ inline uint2& operator-=(uint2& lhs, uint rhs) noexcept { lhs = convert_uint2(lhs - rhs); return lhs; }

constexpr __device__ inline int2 operator< (uint2 lhs, uint rhs) noexcept { return { lhs.x <  rhs, lhs.y <  rhs }; }
constexpr __device__ inline int2 operator<=(uint2 lhs, uint rhs) noexcept { return { lhs.x <= rhs, lhs.y <= rhs }; }
constexpr __device__ inline int2 operator> (uint2 lhs, uint rhs) noexcept { return { lhs.x >  rhs, lhs.y >  rhs }; }
constexpr __device__ inline int2 operator>=(uint2 lhs, uint rhs) noexcept { return { lhs.x >= rhs, lhs.y >= rhs }; }
constexpr __device__ inline int2 operator==(uint2 lhs, uint rhs) noexcept { return { lhs.x == rhs, lhs.y == rhs }; }
constexpr __device__ inline int2 operator!=(uint2 lhs, uint rhs) noexcept { return { lhs.x != rhs, lhs.y != rhs }; }

// short3

// short3 with short3

constexpr __device__ inline int3 operator+(short3 lhs, short3 rhs) noexcept { return { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z }; }
constexpr __device__ inline int3 operator-(short3 lhs, short3 rhs) noexcept { return { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z }; }
constexpr __device__ inline int3 operator*(short3 lhs, short3 rhs) noexcept { return { lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z }; }
constexpr __device__ inline int3 operator/(short3 lhs, short3 rhs) noexcept { return { lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z }; }

constexpr __device__ inline short3& operator+=(short3& lhs, short3 rhs) noexcept { lhs = convert_short3(lhs + rhs); return lhs; }
constexpr __device__ inline short3& operator-=(short3& lhs, short3 rhs) noexcept { lhs = convert_short3(lhs - rhs); return lhs; }

constexpr __device__ inline int3 operator< (short3 lhs, short3 rhs) noexcept { return { lhs.x <  rhs.x, lhs.y <  rhs.y, lhs.y <  rhs.y }; }
constexpr __device__ inline int3 operator<=(short3 lhs, short3 rhs) noexcept { return { lhs.x <= rhs.x, lhs.y <= rhs.y, lhs.y <= rhs.y }; }
constexpr __device__ inline int3 operator> (short3 lhs, short3 rhs) noexcept { return { lhs.x >  rhs.x, lhs.y >  rhs.y, lhs.y >  rhs.y }; }
constexpr __device__ inline int3 operator>=(short3 lhs, short3 rhs) noexcept { return { lhs.x >= rhs.x, lhs.y >= rhs.y, lhs.y >= rhs.y }; }
constexpr __device__ inline int3 operator==(short3 lhs, short3 rhs) noexcept { return { lhs.x == rhs.x, lhs.y == rhs.y, lhs.y == rhs.y }; }
constexpr __device__ inline int3 operator!=(short3 lhs, short3 rhs) noexcept { return { lhs.x != rhs.x, lhs.y != rhs.y, lhs.y != rhs.y,}; }

// short with short3

constexpr __device__ inline int3 operator+(short lhs, short3 rhs) noexcept { return { lhs + rhs.x, lhs + rhs.y, lhs + rhs.z }; }
constexpr __device__ inline int3 operator-(short lhs, short3 rhs) noexcept { return { lhs - rhs.x, lhs - rhs.y, lhs - rhs.z }; }
constexpr __device__ inline int3 operator*(short lhs, short3 rhs) noexcept { return { lhs * rhs.x, lhs * rhs.y, lhs * rhs.z }; }
constexpr __device__ inline int3 operator/(short lhs, short3 rhs) noexcept { return { lhs / rhs.x, lhs / rhs.y, lhs / rhs.z }; }

constexpr __device__ inline int3 operator< (short lhs, short3 rhs) noexcept { return { lhs <  rhs.x, lhs <  rhs.y, lhs <  rhs.y }; }
constexpr __device__ inline int3 operator<=(short lhs, short3 rhs) noexcept { return { lhs <= rhs.x, lhs <= rhs.y, lhs <= rhs.y }; }
constexpr __device__ inline int3 operator> (short lhs, short3 rhs) noexcept { return { lhs >  rhs.x, lhs >  rhs.y, lhs >  rhs.y }; }
constexpr __device__ inline int3 operator>=(short lhs, short3 rhs) noexcept { return { lhs >= rhs.x, lhs >= rhs.y, lhs >= rhs.y }; }
constexpr __device__ inline int3 operator==(short lhs, short3 rhs) noexcept { return { lhs == rhs.x, lhs == rhs.y, lhs == rhs.y,}; }
constexpr __device__ inline int3 operator!=(short lhs, short3 rhs) noexcept { return { lhs != rhs.x, lhs != rhs.y, lhs != rhs.y }; }

// short3 with short

constexpr __device__ inline int3 operator+(short3 lhs, short rhs) noexcept { return { lhs.x + rhs, lhs.y + rhs, lhs.z + rhs }; }
constexpr __device__ inline int3 operator-(short3 lhs, short rhs) noexcept { return { lhs.x - rhs, lhs.y - rhs, lhs.z - rhs }; }
constexpr __device__ inline int3 operator*(short3 lhs, short rhs) noexcept { return { lhs.x * rhs, lhs.y * rhs, lhs.z * rhs }; }
constexpr __device__ inline int3 operator/(short3 lhs, short rhs) noexcept { return { lhs.x / rhs, lhs.y / rhs, lhs.z / rhs }; }

constexpr __device__ inline short3& operator+=(short3& lhs, short rhs) noexcept { lhs = convert_short3(lhs + rhs); return lhs; }
constexpr __device__ inline short3& operator-=(short3& lhs, short rhs) noexcept { lhs = convert_short3(lhs - rhs); return lhs; }

constexpr __device__ inline int3 operator< (short3 lhs, short rhs) noexcept { return { lhs.x <  rhs, lhs.y <  rhs, lhs.y <  rhs }; }
constexpr __device__ inline int3 operator<=(short3 lhs, short rhs) noexcept { return { lhs.x <= rhs, lhs.y <= rhs, lhs.y <= rhs }; }
constexpr __device__ inline int3 operator> (short3 lhs, short rhs) noexcept { return { lhs.x >  rhs, lhs.y >  rhs, lhs.y >  rhs }; }
constexpr __device__ inline int3 operator>=(short3 lhs, short rhs) noexcept { return { lhs.x >= rhs, lhs.y >= rhs, lhs.y >= rhs }; }
constexpr __device__ inline int3 operator==(short3 lhs, short rhs) noexcept { return { lhs.x == rhs, lhs.y == rhs, lhs.y == rhs }; }
constexpr __device__ inline int3 operator!=(short3 lhs, short rhs) noexcept { return { lhs.x != rhs, lhs.y != rhs, lhs.y != rhs }; }

// int3

// int3 with int3

constexpr __device__ inline int3 operator+(int3 lhs, int3 rhs) noexcept { return { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z }; }
constexpr __device__ inline int3 operator-(int3 lhs, int3 rhs) noexcept { return { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z }; }
constexpr __device__ inline int3 operator*(int3 lhs, int3 rhs) noexcept { return { lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z }; }
constexpr __device__ inline int3 operator/(int3 lhs, int3 rhs) noexcept { return { lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z }; }

constexpr __device__ inline int3& operator+=(int3& lhs, int3 rhs) noexcept { lhs = convert_int3(lhs + rhs); return lhs; }
constexpr __device__ inline int3& operator-=(int3& lhs, int3 rhs) noexcept { lhs = convert_int3(lhs - rhs); return lhs; }

constexpr __device__ inline int3 operator< (int3 lhs, int3 rhs) noexcept { return { lhs.x <  rhs.x, lhs.y <  rhs.y, lhs.y <  rhs.y }; }
constexpr __device__ inline int3 operator<=(int3 lhs, int3 rhs) noexcept { return { lhs.x <= rhs.x, lhs.y <= rhs.y, lhs.y <= rhs.y }; }
constexpr __device__ inline int3 operator> (int3 lhs, int3 rhs) noexcept { return { lhs.x >  rhs.x, lhs.y >  rhs.y, lhs.y >  rhs.y }; }
constexpr __device__ inline int3 operator>=(int3 lhs, int3 rhs) noexcept { return { lhs.x >= rhs.x, lhs.y >= rhs.y, lhs.y >= rhs.y }; }
constexpr __device__ inline int3 operator==(int3 lhs, int3 rhs) noexcept { return { lhs.x == rhs.x, lhs.y == rhs.y, lhs.y == rhs.y }; }
constexpr __device__ inline int3 operator!=(int3 lhs, int3 rhs) noexcept { return { lhs.x != rhs.x, lhs.y != rhs.y, lhs.y != rhs.y,}; }

// int with int3

constexpr __device__ inline int3 operator+(int lhs, int3 rhs) noexcept { return { lhs + rhs.x, lhs + rhs.y, lhs + rhs.z }; }
constexpr __device__ inline int3 operator-(int lhs, int3 rhs) noexcept { return { lhs - rhs.x, lhs - rhs.y, lhs - rhs.z }; }
constexpr __device__ inline int3 operator*(int lhs, int3 rhs) noexcept { return { lhs * rhs.x, lhs * rhs.y, lhs * rhs.z }; }
constexpr __device__ inline int3 operator/(int lhs, int3 rhs) noexcept { return { lhs / rhs.x, lhs / rhs.y, lhs / rhs.z }; }

constexpr __device__ inline int3 operator< (int lhs, int3 rhs) noexcept { return { lhs <  rhs.x, lhs <  rhs.y, lhs <  rhs.y }; }
constexpr __device__ inline int3 operator<=(int lhs, int3 rhs) noexcept { return { lhs <= rhs.x, lhs <= rhs.y, lhs <= rhs.y }; }
constexpr __device__ inline int3 operator> (int lhs, int3 rhs) noexcept { return { lhs >  rhs.x, lhs >  rhs.y, lhs >  rhs.y }; }
constexpr __device__ inline int3 operator>=(int lhs, int3 rhs) noexcept { return { lhs >= rhs.x, lhs >= rhs.y, lhs >= rhs.y }; }
constexpr __device__ inline int3 operator==(int lhs, int3 rhs) noexcept { return { lhs == rhs.x, lhs == rhs.y, lhs == rhs.y,}; }
constexpr __device__ inline int3 operator!=(int lhs, int3 rhs) noexcept { return { lhs != rhs.x, lhs != rhs.y, lhs != rhs.y }; }

// int3 with int

constexpr __device__ inline int3 operator+(int3 lhs, int rhs) noexcept { return { lhs.x + rhs, lhs.y + rhs, lhs.z + rhs }; }
constexpr __device__ inline int3 operator-(int3 lhs, int rhs) noexcept { return { lhs.x - rhs, lhs.y - rhs, lhs.z - rhs }; }
constexpr __device__ inline int3 operator*(int3 lhs, int rhs) noexcept { return { lhs.x * rhs, lhs.y * rhs, lhs.z * rhs }; }
constexpr __device__ inline int3 operator/(int3 lhs, int rhs) noexcept { return { lhs.x / rhs, lhs.y / rhs, lhs.z / rhs }; }

constexpr __device__ inline int3& operator+=(int3& lhs, int rhs) noexcept { lhs = convert_int3(lhs + rhs); return lhs; }
constexpr __device__ inline int3& operator-=(int3& lhs, int rhs) noexcept { lhs = convert_int3(lhs - rhs); return lhs; }

constexpr __device__ inline int3 operator< (int3 lhs, int rhs) noexcept { return { lhs.x <  rhs, lhs.y <  rhs, lhs.y <  rhs }; }
constexpr __device__ inline int3 operator<=(int3 lhs, int rhs) noexcept { return { lhs.x <= rhs, lhs.y <= rhs, lhs.y <= rhs }; }
constexpr __device__ inline int3 operator> (int3 lhs, int rhs) noexcept { return { lhs.x >  rhs, lhs.y >  rhs, lhs.y >  rhs }; }
constexpr __device__ inline int3 operator>=(int3 lhs, int rhs) noexcept { return { lhs.x >= rhs, lhs.y >= rhs, lhs.y >= rhs }; }
constexpr __device__ inline int3 operator==(int3 lhs, int rhs) noexcept { return { lhs.x == rhs, lhs.y == rhs, lhs.y == rhs }; }
constexpr __device__ inline int3 operator!=(int3 lhs, int rhs) noexcept { return { lhs.x != rhs, lhs.y != rhs, lhs.y != rhs }; }

// Missing: operators for ushort3, uint3, float3, double3

// int4

// int4 with int4

constexpr __device__ inline int4 operator+(int4 lhs, int4 rhs) noexcept { return { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w }; }
constexpr __device__ inline int4 operator-(int4 lhs, int4 rhs) noexcept { return { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w }; }
constexpr __device__ inline int4 operator*(int4 lhs, int4 rhs) noexcept { return { lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w }; }
constexpr __device__ inline int4 operator/(int4 lhs, int4 rhs) noexcept { return { lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w }; }

constexpr __device__ inline int4& operator+=(int4& lhs, int4 rhs) noexcept { lhs = convert_int4(lhs + rhs); return lhs; }
constexpr __device__ inline int4& operator-=(int4& lhs, int4 rhs) noexcept { lhs = convert_int4(lhs + rhs); return lhs; }

constexpr __device__ inline int4 operator< (int4 lhs, int4 rhs) noexcept { return { lhs.x <  rhs.x, lhs.y <  rhs.y, lhs.y <  rhs.y, lhs.y <  rhs.y }; }
constexpr __device__ inline int4 operator<=(int4 lhs, int4 rhs) noexcept { return { lhs.x <= rhs.x, lhs.y <= rhs.y, lhs.y <= rhs.y, lhs.y <= rhs.y }; }
constexpr __device__ inline int4 operator> (int4 lhs, int4 rhs) noexcept { return { lhs.x >  rhs.x, lhs.y >  rhs.y, lhs.y >  rhs.y, lhs.y >  rhs.y }; }
constexpr __device__ inline int4 operator>=(int4 lhs, int4 rhs) noexcept { return { lhs.x >= rhs.x, lhs.y >= rhs.y, lhs.y >= rhs.y, lhs.y >= rhs.y }; }
constexpr __device__ inline int4 operator==(int4 lhs, int4 rhs) noexcept { return { lhs.x == rhs.x, lhs.y == rhs.y, lhs.y == rhs.y, lhs.y == rhs.y }; }
constexpr __device__ inline int4 operator!=(int4 lhs, int4 rhs) noexcept { return { lhs.x != rhs.x, lhs.y != rhs.y, lhs.y != rhs.y, lhs.y != rhs.y }; }

// int with int4

constexpr __device__ inline int4 operator+(int lhs, int4 rhs) noexcept { return { lhs + rhs.x, lhs + rhs.y, lhs + rhs.z, lhs + rhs.w }; }
constexpr __device__ inline int4 operator-(int lhs, int4 rhs) noexcept { return { lhs - rhs.x, lhs - rhs.y, lhs - rhs.z, lhs - rhs.w }; }
constexpr __device__ inline int4 operator*(int lhs, int4 rhs) noexcept { return { lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w }; }
constexpr __device__ inline int4 operator/(int lhs, int4 rhs) noexcept { return { lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.w }; }

constexpr __device__ inline int4 operator< (int lhs, int4 rhs) noexcept { return { lhs <  rhs.x, lhs <  rhs.y, lhs <  rhs.y, lhs <  rhs.y }; }
constexpr __device__ inline int4 operator<=(int lhs, int4 rhs) noexcept { return { lhs <= rhs.x, lhs <= rhs.y, lhs <= rhs.y, lhs <= rhs.y }; }
constexpr __device__ inline int4 operator> (int lhs, int4 rhs) noexcept { return { lhs >  rhs.x, lhs >  rhs.y, lhs >  rhs.y, lhs >  rhs.y }; }
constexpr __device__ inline int4 operator>=(int lhs, int4 rhs) noexcept { return { lhs >= rhs.x, lhs >= rhs.y, lhs >= rhs.y, lhs >= rhs.y }; }
constexpr __device__ inline int4 operator==(int lhs, int4 rhs) noexcept { return { lhs == rhs.x, lhs == rhs.y, lhs == rhs.y, lhs == rhs.y }; }
constexpr __device__ inline int4 operator!=(int lhs, int4 rhs) noexcept { return { lhs != rhs.x, lhs != rhs.y, lhs != rhs.y, lhs != rhs.y }; }

// int4 with int

constexpr __device__ inline int4 operator+(int4 lhs, int rhs) noexcept { return { lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs }; }
constexpr __device__ inline int4 operator-(int4 lhs, int rhs) noexcept { return { lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs }; }
constexpr __device__ inline int4 operator*(int4 lhs, int rhs) noexcept { return { lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs }; }
constexpr __device__ inline int4 operator/(int4 lhs, int rhs) noexcept { return { lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs }; }

constexpr __device__ inline int4& operator+=(int4& lhs, int rhs) noexcept { lhs = convert_int4(lhs + rhs); return lhs; }
constexpr __device__ inline int4& operator-=(int4& lhs, int rhs) noexcept { lhs = convert_int4(lhs + rhs); return lhs; }

constexpr __device__ inline int4 operator< (int4 lhs, int rhs) noexcept { return { lhs.x <  rhs, lhs.y <  rhs, lhs.y <  rhs, lhs.y <  rhs }; }
constexpr __device__ inline int4 operator<=(int4 lhs, int rhs) noexcept { return { lhs.x <= rhs, lhs.y <= rhs, lhs.y <= rhs, lhs.y <= rhs }; }
constexpr __device__ inline int4 operator> (int4 lhs, int rhs) noexcept { return { lhs.x >  rhs, lhs.y >  rhs, lhs.y >  rhs, lhs.y >  rhs }; }
constexpr __device__ inline int4 operator>=(int4 lhs, int rhs) noexcept { return { lhs.x >= rhs, lhs.y >= rhs, lhs.y >= rhs, lhs.y >= rhs }; }
constexpr __device__ inline int4 operator==(int4 lhs, int rhs) noexcept { return { lhs.x == rhs, lhs.y == rhs, lhs.y == rhs, lhs.y == rhs }; }
constexpr __device__ inline int4 operator!=(int4 lhs, int rhs) noexcept { return { lhs.x != rhs, lhs.y != rhs, lhs.y != rhs, lhs.y != rhs }; }

// float2

// float2 with float2

constexpr __device__ inline float2 operator+(float2 lhs, float2 rhs) noexcept { return { lhs.x + rhs.x, lhs.y + rhs.y }; }
constexpr __device__ inline float2 operator-(float2 lhs, float2 rhs) noexcept { return { lhs.x - rhs.x, lhs.y - rhs.y }; }
constexpr __device__ inline float2 operator*(float2 lhs, float2 rhs) noexcept { return { lhs.x * rhs.x, lhs.y * rhs.y }; }
constexpr __device__ inline float2 operator/(float2 lhs, float2 rhs) noexcept { return { lhs.x / rhs.x, lhs.y / rhs.y }; }

constexpr __device__ inline float2& operator+=(float2& lhs, float2 rhs) noexcept { lhs = lhs + rhs; return lhs; }
constexpr __device__ inline float2& operator-=(float2& lhs, float2 rhs) noexcept { lhs = lhs - rhs; return lhs; }

constexpr __device__ inline int2 operator< (float2 lhs, float2 rhs) noexcept { return { lhs.x <  rhs.x, lhs.y <  rhs.y }; }
constexpr __device__ inline int2 operator<=(float2 lhs, float2 rhs) noexcept { return { lhs.x <= rhs.x, lhs.y <= rhs.y }; }
constexpr __device__ inline int2 operator> (float2 lhs, float2 rhs) noexcept { return { lhs.x >  rhs.x, lhs.y >  rhs.y }; }
constexpr __device__ inline int2 operator>=(float2 lhs, float2 rhs) noexcept { return { lhs.x >= rhs.x, lhs.y >= rhs.y }; }
constexpr __device__ inline int2 operator==(float2 lhs, float2 rhs) noexcept { return { lhs.x == rhs.x, lhs.y == rhs.y }; }
constexpr __device__ inline int2 operator!=(float2 lhs, float2 rhs) noexcept { return { lhs.x != rhs.x, lhs.y != rhs.y }; }

// float with float2

constexpr __device__ inline float2 operator+(float lhs, float2 rhs) noexcept { return { lhs + rhs.x, lhs + rhs.y }; }
constexpr __device__ inline float2 operator-(float lhs, float2 rhs) noexcept { return { lhs - rhs.x, lhs - rhs.y }; }
constexpr __device__ inline float2 operator*(float lhs, float2 rhs) noexcept { return { lhs * rhs.x, lhs * rhs.y }; }
constexpr __device__ inline float2 operator/(float lhs, float2 rhs) noexcept { return { lhs / rhs.x, lhs / rhs.y }; }

constexpr __device__ inline int2 operator< (float lhs, float2 rhs) noexcept { return { lhs <  rhs.x, lhs <  rhs.y }; }
constexpr __device__ inline int2 operator<=(float lhs, float2 rhs) noexcept { return { lhs <= rhs.x, lhs <= rhs.y }; }
constexpr __device__ inline int2 operator> (float lhs, float2 rhs) noexcept { return { lhs >  rhs.x, lhs >  rhs.y }; }
constexpr __device__ inline int2 operator>=(float lhs, float2 rhs) noexcept { return { lhs >= rhs.x, lhs >= rhs.y }; }
constexpr __device__ inline int2 operator==(float lhs, float2 rhs) noexcept { return { lhs == rhs.x, lhs == rhs.y }; }
constexpr __device__ inline int2 operator!=(float lhs, float2 rhs) noexcept { return { lhs != rhs.x, lhs != rhs.y }; }

// float2 with float

constexpr __device__ inline float2 operator+(float2 lhs, float rhs) noexcept { return { lhs.x + rhs, lhs.y + rhs }; }
constexpr __device__ inline float2 operator-(float2 lhs, float rhs) noexcept { return { lhs.x - rhs, lhs.y - rhs }; }
constexpr __device__ inline float2 operator*(float2 lhs, float rhs) noexcept { return { lhs.x * rhs, lhs.y * rhs }; }
constexpr __device__ inline float2 operator/(float2 lhs, float rhs) noexcept { return { lhs.x / rhs, lhs.y / rhs }; }

constexpr __device__ inline float2& operator+=(float2& lhs, float rhs) noexcept { lhs = lhs + rhs; return lhs; }
constexpr __device__ inline float2& operator-=(float2& lhs, float rhs) noexcept { lhs = lhs - rhs; return lhs; }

constexpr __device__ inline int2 operator< (float2 lhs, float rhs) noexcept { return { lhs.x <  rhs, lhs.y <  rhs }; }
constexpr __device__ inline int2 operator<=(float2 lhs, float rhs) noexcept { return { lhs.x <= rhs, lhs.y <= rhs }; }
constexpr __device__ inline int2 operator> (float2 lhs, float rhs) noexcept { return { lhs.x >  rhs, lhs.y >  rhs }; }
constexpr __device__ inline int2 operator>=(float2 lhs, float rhs) noexcept { return { lhs.x >= rhs, lhs.y >= rhs }; }
constexpr __device__ inline int2 operator==(float2 lhs, float rhs) noexcept { return { lhs.x == rhs, lhs.y == rhs }; }
constexpr __device__ inline int2 operator!=(float2 lhs, float rhs) noexcept { return { lhs.x != rhs, lhs.y != rhs }; }

// float4

// float4 with float4

constexpr __device__ inline float4 operator+(float4 lhs, float4 rhs) noexcept { return { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w }; }
constexpr __device__ inline float4 operator-(float4 lhs, float4 rhs) noexcept { return { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w }; }
constexpr __device__ inline float4 operator*(float4 lhs, float4 rhs) noexcept { return { lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w }; }
constexpr __device__ inline float4 operator/(float4 lhs, float4 rhs) noexcept { return { lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w }; }

constexpr __device__ inline float4& operator+=(float4& lhs, float4 rhs) noexcept { lhs = lhs + rhs; return lhs; }
constexpr __device__ inline float4& operator-=(float4& lhs, float4 rhs) noexcept { lhs = lhs + rhs; return lhs; }

constexpr __device__ inline int4 operator< (float4 lhs, float4 rhs) noexcept { return { lhs.x <  rhs.x, lhs.y <  rhs.y, lhs.y <  rhs.y, lhs.y <  rhs.y }; }
constexpr __device__ inline int4 operator<=(float4 lhs, float4 rhs) noexcept { return { lhs.x <= rhs.x, lhs.y <= rhs.y, lhs.y <= rhs.y, lhs.y <= rhs.y }; }
constexpr __device__ inline int4 operator> (float4 lhs, float4 rhs) noexcept { return { lhs.x >  rhs.x, lhs.y >  rhs.y, lhs.y >  rhs.y, lhs.y >  rhs.y }; }
constexpr __device__ inline int4 operator>=(float4 lhs, float4 rhs) noexcept { return { lhs.x >= rhs.x, lhs.y >= rhs.y, lhs.y >= rhs.y, lhs.y >= rhs.y }; }
constexpr __device__ inline int4 operator==(float4 lhs, float4 rhs) noexcept { return { lhs.x == rhs.x, lhs.y == rhs.y, lhs.y == rhs.y, lhs.y == rhs.y }; }
constexpr __device__ inline int4 operator!=(float4 lhs, float4 rhs) noexcept { return { lhs.x != rhs.x, lhs.y != rhs.y, lhs.y != rhs.y, lhs.y != rhs.y }; }

// float with float4

constexpr __device__ inline float4 operator+(float lhs, float4 rhs) noexcept { return { lhs + rhs.x, lhs + rhs.y, lhs + rhs.z, lhs + rhs.w }; }
constexpr __device__ inline float4 operator-(float lhs, float4 rhs) noexcept { return { lhs - rhs.x, lhs - rhs.y, lhs - rhs.z, lhs - rhs.w }; }
constexpr __device__ inline float4 operator*(float lhs, float4 rhs) noexcept { return { lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w }; }
constexpr __device__ inline float4 operator/(float lhs, float4 rhs) noexcept { return { lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.w }; }

constexpr __device__ inline int4 operator< (float lhs, float4 rhs) noexcept { return { lhs <  rhs.x, lhs <  rhs.y, lhs <  rhs.y, lhs <  rhs.y }; }
constexpr __device__ inline int4 operator<=(float lhs, float4 rhs) noexcept { return { lhs <= rhs.x, lhs <= rhs.y, lhs <= rhs.y, lhs <= rhs.y }; }
constexpr __device__ inline int4 operator> (float lhs, float4 rhs) noexcept { return { lhs >  rhs.x, lhs >  rhs.y, lhs >  rhs.y, lhs >  rhs.y }; }
constexpr __device__ inline int4 operator>=(float lhs, float4 rhs) noexcept { return { lhs >= rhs.x, lhs >= rhs.y, lhs >= rhs.y, lhs >= rhs.y }; }
constexpr __device__ inline int4 operator==(float lhs, float4 rhs) noexcept { return { lhs == rhs.x, lhs == rhs.y, lhs == rhs.y, lhs == rhs.y }; }
constexpr __device__ inline int4 operator!=(float lhs, float4 rhs) noexcept { return { lhs != rhs.x, lhs != rhs.y, lhs != rhs.y, lhs != rhs.y }; }

// float4 with float

constexpr __device__ inline float4 operator+(float4 lhs, float rhs) noexcept { return { lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs }; }
constexpr __device__ inline float4 operator-(float4 lhs, float rhs) noexcept { return { lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs }; }
constexpr __device__ inline float4 operator*(float4 lhs, float rhs) noexcept { return { lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs }; }
constexpr __device__ inline float4 operator/(float4 lhs, float rhs) noexcept { return { lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs }; }

constexpr __device__ inline float4& operator+=(float4& lhs, float rhs) noexcept { lhs = lhs + rhs; return lhs; }
constexpr __device__ inline float4& operator-=(float4& lhs, float rhs) noexcept { lhs = lhs + rhs; return lhs; }

constexpr __device__ inline int4 operator< (float4 lhs, float rhs) noexcept { return { lhs.x <  rhs, lhs.y <  rhs, lhs.y <  rhs, lhs.y <  rhs }; }
constexpr __device__ inline int4 operator<=(float4 lhs, float rhs) noexcept { return { lhs.x <= rhs, lhs.y <= rhs, lhs.y <= rhs, lhs.y <= rhs }; }
constexpr __device__ inline int4 operator> (float4 lhs, float rhs) noexcept { return { lhs.x >  rhs, lhs.y >  rhs, lhs.y >  rhs, lhs.y >  rhs }; }
constexpr __device__ inline int4 operator>=(float4 lhs, float rhs) noexcept { return { lhs.x >= rhs, lhs.y >= rhs, lhs.y >= rhs, lhs.y >= rhs }; }
constexpr __device__ inline int4 operator==(float4 lhs, float rhs) noexcept { return { lhs.x == rhs, lhs.y == rhs, lhs.y == rhs, lhs.y == rhs }; }
constexpr __device__ inline int4 operator!=(float4 lhs, float rhs) noexcept { return { lhs.x != rhs, lhs.y != rhs, lhs.y != rhs, lhs.y != rhs }; }

#endif // VECTORIZED_TYPES_BASIC_OPERATORS

// float4 with array of 4 floats

constexpr __device__ inline float4 as_float4(float const(& floats)[4]) noexcept
{
    return { floats[0], floats[1], floats[2], floats[3] };
}

// array of 4 floats with float4

typedef float float_array4[4];

__device__ inline float_array4& as_float_array(float4& floats) noexcept
{
    return reinterpret_cast<float_array4 &>(floats);
}

constexpr __device__ inline float4 operator+(float_array4& lhs, float4 rhs) noexcept { float4 lhs_ = as_float4(lhs); return lhs_ + rhs; }
constexpr __device__ inline float4 operator-(float_array4& lhs, float4 rhs) noexcept { float4 lhs_ = as_float4(lhs); return lhs_ - rhs; }
constexpr __device__ inline float4 operator*(float_array4& lhs, float4 rhs) noexcept { float4 lhs_ = as_float4(lhs); return lhs_ * rhs; }
constexpr __device__ inline float4 operator/(float_array4& lhs, float4 rhs) noexcept { float4 lhs_ = as_float4(lhs); return lhs_ / rhs; }

constexpr __device__ inline float_array4& operator+=(float_array4& lhs, float4 rhs) noexcept
{
    lhs[0] += rhs.x;
    lhs[1] += rhs.y;
    lhs[2] += rhs.z;
    lhs[3] += rhs.w;
    return lhs;
}

constexpr __device__ inline float_array4& operator-=(float_array4& lhs, float4 rhs) noexcept
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

//constexpr long lmin(long x, long y) noexcept { return x < y ? x : y; }
//constexpr long lumin(ulong x, ulong y) noexcept { return x < y ? x : y; }

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

__forceinline__ __device__ void syncWarp()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
    __syncwarp();
#endif
}


// Atomic operations
// Add
__device__ inline int atomic_add(int *address, int val) { return atomicAdd(address, val); };
__device__ inline uint atomic_add(uint *address, uint val) { return atomicAdd(address, val); };

// Sub
__device__ inline int atomic_sub(int *address, int val) { return atomicSub(address, val); };
__device__ inline uint atomic_sub(uint *address, uint val) { return atomicSub(address, val); };

// Xchg
__device__ inline int atomic_xchg(int *address, int val) { return atomicExch(address, val); };
__device__ inline uint atomic_xchg(uint *address, uint val) { return atomicExch(address, val); };

// Inc
__device__ inline int atomic_inc(int *address) { return atomicAdd(address, 1); };
__device__ inline uint atomic_inc(uint *address) { return atomicAdd(address, 1); };

// Dec
__device__ inline int atomic_dec(int *address) { return atomicSub(address, 1); };
__device__ inline uint atomic_dec(uint *address) { return atomicSub(address, 1); };

// Cmpxchg
__device__ inline int atomic_cmpxchg(int *address, int cmp, int val) { return atomicCAS(address, cmp, val); };
__device__ inline uint atomic_cmpxchg(uint *address, uint cmp, uint val) { return atomicCAS(address, cmp, val); };

// Min
__device__ inline int atomic_min(int *address, int val) { return atomicMin(address, val); };
__device__ inline uint atomic_min(uint *address, uint val) { return atomicMin(address, val); };

// Max
__device__ inline int atomic_max(int *address, int val) { return atomicMax(address, val); };
__device__ inline uint atomic_max(uint *address, uint val) { return atomicMax(address, val); };

// And
__device__ inline int atomic_and(int *address, int val) { return atomicAnd(address, val); };
__device__ inline uint atomic_and(uint *address, uint val) { return atomicAnd(address, val); };

// Or
__device__ inline int atomic_or(int *address, int val) { return atomicOr(address, val); };
__device__ inline uint atomic_or(uint *address, uint val) { return atomicOr(address, val); };

// Xor
__device__ inline int atomic_xor(int *address, int val) { return atomicXor(address, val); };
__device__ inline uint atomic_xor(uint *address, uint val) { return atomicXor(address, val); };

#endif // __OPENCL_VERSION__
#endif // PORT_FROM_OPENCL_VECTOR_TYPES_CUH_
