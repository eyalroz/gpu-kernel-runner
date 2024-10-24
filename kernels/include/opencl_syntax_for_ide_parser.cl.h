/**
 * @file opencl_syntax_for_ide_parser.cl.h
 *
 * @brief Conditionally-compiled definitions which let some IDE parsers -
 * currently JetBrains CLion and Eclipse CDT - better "accept" OpenCL
 * sources without a specialized plugin.
 *
 * @copyright (c) 2020-2022, GE Healthcare.
 * @copyright (c) 2022, Eyal Rozenberg.
 *
 * @license BSD 3-clause license; see the `LICENSE` file or
 * @url https://opensource.org/licenses/BSD-3-Clause
 *
 * @note This is not a complete set of definitions.
 */
#ifndef OPENCL_SYNTAX_FOR_IDE_PARSER_CL_H_
#define OPENCL_SYNTAX_FOR_IDE_PARSER_CL_H_

#if defined(__CDT_PARSER__) || defined (__JETBRAINS_IDE__)
#ifndef __CUDA_ARCH__

#ifndef CONCAT_IDENTIFIERS
#define CONCAT_IDENTIFIERS_INNER(_token1, _token2) _token1 ## _token2
#define CONCAT_IDENTIFIERS(_token1, _token2) CONCAT_IDENTIFIERS_INNER(_token1, _token2)
#endif // CONCAT_IDENTIFIERS

#define __kernel
#define __global
#define __private
#define __constant const
#define __local
#define __kernel

#ifndef restrict
#define restrict
#define __restrict
#endif // restrict

#ifndef UNSIGNED_INTEGRAL_SHORTHANDS_DEFINED
typedef unsigned char  uchar;
typedef unsigned short ushort;
typedef unsigned int   uint;
typedef unsigned long  ulong;
#endif

typedef unsigned long  size_t;
#if !defined(__CUDA_FP16_H__) && !defined(PORT_FROM_OPENCL_ENABLE_HALF_PRECISION)
typedef float half; // well, not really, but for syntax purposes this works I guess
#endif

#define TYPEDEF_STRUCT_OF_2(_tp) struct CONCAT_IDENTIFIERS(_tp, 2) { _tp x, y; }
TYPEDEF_STRUCT_OF_2(char);
TYPEDEF_STRUCT_OF_2(uchar);
TYPEDEF_STRUCT_OF_2(short);
TYPEDEF_STRUCT_OF_2(ushort);
TYPEDEF_STRUCT_OF_2(int);
TYPEDEF_STRUCT_OF_2(uint);
TYPEDEF_STRUCT_OF_2(long);
TYPEDEF_STRUCT_OF_2(ulong);
TYPEDEF_STRUCT_OF_2(half);
TYPEDEF_STRUCT_OF_2(float);
TYPEDEF_STRUCT_OF_2(double);

#define TYPEDEF_STRUCT_OF_4(_tp) struct CONCAT_IDENTIFIERS(_tp, 4) { _tp x, y, z, w; }
TYPEDEF_STRUCT_OF_4(char);
TYPEDEF_STRUCT_OF_4(uchar);
TYPEDEF_STRUCT_OF_4(short);
TYPEDEF_STRUCT_OF_4(ushort);
TYPEDEF_STRUCT_OF_4(int);
TYPEDEF_STRUCT_OF_4(uint);
TYPEDEF_STRUCT_OF_4(long);
TYPEDEF_STRUCT_OF_4(ulong);
TYPEDEF_STRUCT_OF_4(half);
TYPEDEF_STRUCT_OF_4(float);
TYPEDEF_STRUCT_OF_4(double);

//
//struct short4 { short x, y, z, w; };
//struct ushort4 { unsigned short x, y, z, w; };
//struct int4 { int x, y, z, w; };
//struct uint4{ unsigned int x, y, z, w; };
//struct float4 { float x, y, z, w; };
//struct half4 { half x, y, z, w; };

#define convert_int(_x) (int) (_x)
#define convert_float(_x) (float) (_x)

#define ceil(x) (x)
#define floor(x) (x)
#define fmax(x, y) (x)
#define fmin(x, y) (x)

#define barrier(...)
#define native_recip(x) (x)

inline float2 vload2(size_t offset, const float* p);
inline void vstore2(const float2& value, size_t offset, float* p);
inline float4 vload4(size_t offset, const float* p);
inline void vstore4(const float4& value, size_t offset, float* p);

double asin(double x);
float asinf(float x);

#ifndef max
#define max(x,y) x
#define min(x,y) x
#endif

#ifndef barrier
#define barrier(_x) do {} while(0)
#endif

uint get_work_dim();
size_t get_global_size(uint dimindx);
size_t get_global_id(uint dimindx);
size_t get_local_size(uint dimindx);
size_t get_enqueued_local_size( uint dimindx);
size_t get_local_id(uint dimindx);
size_t get_num_groups(uint dimindx);
size_t get_group_id(uint dimindx);
size_t get_global_offset(uint dimindx);
size_t get_global_linear_id();
size_t get_local_linear_id();

int printf (const char * restrict format, ... );

uint convert_uint(float x);

int atomic_cmpxchg (volatile __global int *p , int cmp, int val);
unsigned int atomic_cmpxchg (volatile __global unsigned int *p , unsigned int cmp, unsigned int val);
int atomic_cmpxchg (volatile __local int *p , int cmp, int val);
unsigned int atomic_cmpxchg (volatile __local unsigned int *p , unsigned int cmp, unsigned int val);

float select(float on_false, float on_true, int selector);

// float2 with float2

inline float2 operator+(float2 lhs, float2 rhs) noexcept { return { lhs.x + rhs.x, lhs.y + rhs.y }; }
inline float2 operator-(float2 lhs, float2 rhs) noexcept { return { lhs.x - rhs.x, lhs.y - rhs.y }; }
inline float2 operator*(float2 lhs, float2 rhs) noexcept { return { lhs.x * rhs.x, lhs.y * rhs.y }; }
inline float2 operator/(float2 lhs, float2 rhs) noexcept { return { lhs.x / rhs.x, lhs.y / rhs.y }; }

inline float2& operator+=(float2& lhs, float2 rhs) noexcept { lhs = lhs + rhs; return lhs; }
inline float2& operator-=(float2& lhs, float2 rhs) noexcept { lhs = lhs - rhs; return lhs; }

inline bool operator==(float2 lhs, float2 rhs) noexcept { return lhs.x == rhs.x && lhs.y == rhs.y; }
inline bool operator!=(float2 lhs, float2 rhs) noexcept { return lhs.x != rhs.x || lhs.y != rhs.y; }

// float with float2

inline float2 operator+(float lhs, float2 rhs) noexcept { return { lhs + rhs.x, lhs + rhs.y }; }
inline float2 operator-(float lhs, float2 rhs) noexcept { return { lhs - rhs.x, lhs - rhs.y }; }
inline float2 operator*(float lhs, float2 rhs) noexcept { return { lhs * rhs.x, lhs * rhs.y }; }
inline float2 operator/(float lhs, float2 rhs) noexcept { return { lhs / rhs.x, lhs / rhs.y }; }

// float2 with float

inline float2 operator+(float2 lhs, float rhs) noexcept { return { lhs.x + rhs, lhs.y + rhs }; }
inline float2 operator-(float2 lhs, float rhs) noexcept { return { lhs.x - rhs, lhs.y - rhs }; }
inline float2 operator*(float2 lhs, float rhs) noexcept { return { lhs.x * rhs, lhs.y * rhs }; }
inline float2 operator/(float2 lhs, float rhs) noexcept { return { lhs.x / rhs, lhs.y / rhs }; }

inline float2& operator+=(float2& lhs, float rhs) noexcept { lhs = lhs + rhs; return lhs; }
inline float2& operator-=(float2& lhs, float rhs) noexcept { lhs = lhs - rhs; return lhs; }

// float4 with float4

inline float4 operator+(float4 lhs, float4 rhs) noexcept { return { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w }; }
inline float4 operator-(float4 lhs, float4 rhs) noexcept { return { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w }; }
inline float4 operator*(float4 lhs, float4 rhs) noexcept { return { lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w }; }
inline float4 operator/(float4 lhs, float4 rhs) noexcept { return { lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w }; }

inline float4& operator+=(float4& lhs, float4 rhs) noexcept { lhs = lhs + rhs; return lhs; }
inline float4& operator-=(float4& lhs, float4 rhs) noexcept { lhs = lhs + rhs; return lhs; }

inline bool operator==(float4 lhs, float4 rhs) noexcept { return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w; }
inline bool operator!=(float4 lhs, float4 rhs) noexcept { return lhs.x != rhs.x || lhs.y != rhs.y || lhs.z != rhs.z || lhs.w != rhs.w; }

// float with float4

inline float4 operator+(float lhs, float4 rhs) noexcept { return { lhs + rhs.x, lhs + rhs.y, lhs + rhs.z, lhs + rhs.w }; }
inline float4 operator-(float lhs, float4 rhs) noexcept { return { lhs - rhs.x, lhs - rhs.y, lhs - rhs.z, lhs - rhs.w }; }
inline float4 operator*(float lhs, float4 rhs) noexcept { return { lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w }; }
inline float4 operator/(float lhs, float4 rhs) noexcept { return { lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.w }; }

// float4 with float

inline float4 operator+(float4 lhs, float rhs) noexcept { return { lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs }; }
inline float4 operator-(float4 lhs, float rhs) noexcept { return { lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs }; }
inline float4 operator*(float4 lhs, float rhs) noexcept { return { lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs }; }
inline float4 operator/(float4 lhs, float rhs) noexcept { return { lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs }; }

inline float4& operator+=(float4& lhs, float rhs) noexcept { lhs = lhs + rhs; return lhs; }
inline float4& operator-=(float4& lhs, float rhs) noexcept { lhs = lhs + rhs; return lhs; }

// ---- half2 and half4 ops

// half2 with half2

//inline half2 operator+(half2 lhs, half2 rhs) noexcept { return { lhs.x + rhs.x, lhs.y + rhs.y }; }
//inline half2 operator-(half2 lhs, half2 rhs) noexcept { return { lhs.x - rhs.x, lhs.y - rhs.y }; }
//inline half2 operator*(half2 lhs, half2 rhs) noexcept { return { lhs.x * rhs.x, lhs.y * rhs.y }; }
//inline half2 operator/(half2 lhs, half2 rhs) noexcept { return { lhs.x / rhs.x, lhs.y / rhs.y }; }
//
//inline half2& operator+=(half2& lhs, half2 rhs) noexcept { lhs = lhs + rhs; return lhs; }
//inline half2& operator-=(half2& lhs, half2 rhs) noexcept { lhs = lhs - rhs; return lhs; }
//
//inline bool operator==(half2 lhs, half2 rhs) noexcept { return lhs.x == rhs.x && lhs.y == rhs.y; }
//inline bool operator!=(half2 lhs, half2 rhs) noexcept { return lhs.x != rhs.x || lhs.y != rhs.y; }

// half with half2

inline half2 operator+(half lhs, half2 rhs) noexcept { return { lhs + rhs.x, lhs + rhs.y }; }
inline half2 operator-(half lhs, half2 rhs) noexcept { return { lhs - rhs.x, lhs - rhs.y }; }
inline half2 operator*(half lhs, half2 rhs) noexcept { return { lhs * rhs.x, lhs * rhs.y }; }
inline half2 operator/(half lhs, half2 rhs) noexcept { return { lhs / rhs.x, lhs / rhs.y }; }

// half2 with half

inline half2 operator+(half2 lhs, half rhs) noexcept { return { lhs.x + rhs, lhs.y + rhs }; }
inline half2 operator-(half2 lhs, half rhs) noexcept { return { lhs.x - rhs, lhs.y - rhs }; }
inline half2 operator*(half2 lhs, half rhs) noexcept { return { lhs.x * rhs, lhs.y * rhs }; }
inline half2 operator/(half2 lhs, half rhs) noexcept { return { lhs.x / rhs, lhs.y / rhs }; }

inline half2& operator+=(half2& lhs, half rhs) noexcept { lhs = lhs + rhs; return lhs; }
inline half2& operator-=(half2& lhs, half rhs) noexcept { lhs = lhs - rhs; return lhs; }

// half4 with half4

inline half4 operator+(half4 lhs, half4 rhs) noexcept { return { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w }; }
inline half4 operator-(half4 lhs, half4 rhs) noexcept { return { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w }; }
inline half4 operator*(half4 lhs, half4 rhs) noexcept { return { lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w }; }
inline half4 operator/(half4 lhs, half4 rhs) noexcept { return { lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w }; }

inline half4& operator+=(half4& lhs, half4 rhs) noexcept { lhs = lhs + rhs; return lhs; }
inline half4& operator-=(half4& lhs, half4 rhs) noexcept { lhs = lhs + rhs; return lhs; }

inline bool operator==(half4 lhs, half4 rhs) noexcept { return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w; }
inline bool operator!=(half4 lhs, half4 rhs) noexcept { return lhs.x != rhs.x || lhs.y != rhs.y || lhs.z != rhs.z || lhs.w != rhs.w; }

// half with half4

inline half4 operator+(half lhs, half4 rhs) noexcept { return { lhs + rhs.x, lhs + rhs.y, lhs + rhs.z, lhs + rhs.w }; }
inline half4 operator-(half lhs, half4 rhs) noexcept { return { lhs - rhs.x, lhs - rhs.y, lhs - rhs.z, lhs - rhs.w }; }
inline half4 operator*(half lhs, half4 rhs) noexcept { return { lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w }; }
inline half4 operator/(half lhs, half4 rhs) noexcept { return { lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.w }; }

// half4 with half

inline half4 operator+(half4 lhs, half rhs) noexcept { return { lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs }; }
inline half4 operator-(half4 lhs, half rhs) noexcept { return { lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs }; }
inline half4 operator*(half4 lhs, half rhs) noexcept { return { lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs }; }
inline half4 operator/(half4 lhs, half rhs) noexcept { return { lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs }; }

inline half4& operator+=(half4& lhs, half rhs) noexcept { lhs = lhs + rhs; return lhs; }
inline half4& operator-=(half4& lhs, half rhs) noexcept { lhs = lhs + rhs; return lhs; }

#endif // __CUDA_ARCH__

#endif // defined(__CDT_PARSER__) || defined (__JETBRAINS_IDE__)

#endif // OPENCL_SYNTAX_FOR_IDE_PARSER_CL_H_
