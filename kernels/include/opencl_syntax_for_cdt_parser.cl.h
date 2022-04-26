/**
 * @file opencl_syntax_for_cdt_parser.cl.h
 *
 * @brief Conditionally-compiled definitions which let CDT's parser better 
 * "accept" OpenCL sources without a specialized plugin.
 *
 * @note This is not a complete set of definitions.
 */
#ifndef OPENCL_SYNTAX_FOR_CDT_PARSER_CL_H_
#define OPENCL_SYNTAX_FOR_CDT_PARSER_CL_H_

#ifdef __CDT_PARSER__

#define __kernel
#define __global
#define __private
#define __constant const

struct short2 { short x, y; };
struct ushort2 { unsigned short x, y; };
struct int2 { int x, y; };
struct uint2 { unsigned int x, y; };
struct float2 { float x, y; };

struct short4 { short x, y, z, w; };
struct ushort4 { unsigned short x, y, z, w; };
struct int4 { int x, y, z, w; };
struct uint4{ unsigned x, y, z, w; };
struct float4 { float x, y, z, w; };

#define uchar unsigned char
#define ushort unsigned short
#define uint unsigned int
#define ulong unsigned long
#define size_t unsigned long

#define convert_int(_x) (int) (_x)
#define convert_float(_x) (float) (_x)

#define ceil(x) (x)
#define floor(x) (x)
#define fmax(x, y) (x)
#define fmin(x, y) (x)

#define barrier()
#define native_recip(x) (x)

inline float2 vload2(size_t offset, const float* p);
inline void vstore2(const float2& value, size_t offset, float* p);
inline float4 vload4(size_t offset, const float* p);
inline void vstore4(const float4& value, size_t offset, float* p);

double asin(double x);
float asinf(float x);

#define max(x,y) x
#define min(x,y) x

#define barrier(_x)

#define __local
#define __kernel
#define restrict

typedef uchar  uint8_t;
typedef ushort uint16_t;
typedef uint   uint32_t;
typedef ulong  uint64_t;

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

#endif // __CDT_PARSER__

#endif // OPENCL_SYNTAX_FOR_CDT_PARSER_CL_H_
