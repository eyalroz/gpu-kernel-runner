/**
 * @file
 *
 * @brief 
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
#ifndef PORT_FROM_OPENCL_DEFINES_CUH_
#define PORT_FROM_OPENCL_DEFINES_CUH_

#ifndef __OPENCL_VERSION__

// §6.12. Preprocessor Directives and Macros
// ------------------------------------------

// TODO: Define __FAST_RELAXED_MATH__ if the CUDA compiler is in that mode

// See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation
#define __ENDIAN_LITTLE__ 1

// Not defining OpenCL's __ROUNDING_MODE__, it is deprecated

// §6.15.2.1
// --------------------------------------

//#ifdef PORT_FROM_OPENCL_ENABLE_MATH_DEFINES
#define FLT_DIG          6
#define FLT_MANT_DIG     24
#define FLT_MAX_10_EXP   +38
#define FLT_MAX_EXP      +128
#define FLT_MIN_10_EXP   -37
#define FLT_MIN_EXP      -125
#define FLT_RADIX        2
#define FLT_MAX          0x1.fffffep127f
#define FLT_MIN          0x1.0p-126f
#define FLT_EPSILON      0x1.0p-23f

#define M_E_F           2.71828174591064f
#define M_LOG2E_F       1.44269502162933f
#define M_LOG10E_F      0.43429449200630f
#define M_LN2_F         0.69314718246460f
#define M_LN10_F        2.30258512496948f
#define M_PI_F          3.14159274101257f
#define M_PI_2_F        1.57079637050629f
#define M_PI_4_F        0.78539818525314f
#define M_1_PI_F        0.31830987334251f
#define M_2_PI_F        0.63661974668503f
#define M_2_SQRTPI_F    1.12837922573090f
#define M_SQRT2_F       1.41421353816986f
#define M_SQRT1_2_F     0.70710676908493f

#define DBL_DIG          15
#define DBL_MANT_DIG     53
#define DBL_MAX_10_EXP   +308
#define DBL_MAX_EXP      +1024
#define DBL_MIN_10_EXP   -307
#define DBL_MIN_EXP      -1021
#define DBL_RADIX        2
#define DBL_MAX          0x1.fffffffffffffp1023
#define DBL_MIN          0x1.0p-1022
#define DBL_EPSILON      0x1.0p-52

#define M_E             2.718281828459045090796
#define M_LOG2E         1.442695040888963387005
#define M_LOG10E        0.434294481903251816668
#define M_LN2           0.693147180559945286227
#define M_LN10          2.302585092994045901094
#define M_PI            3.141592653589793115998
#define M_PI_2          1.570796326794896557999
#define M_PI_4          0.785398163397448278999
#define M_1_PI          0.318309886183790691216
#define M_2_PI          0.636619772367581382433
#define M_2_SQRTPI      1.128379167095512558561
#define M_SQRT2         1.414213562373095145475
#define M_SQRT1_2       0.707106781186547572737

#define HALF_DIG            3
#define HALF_MANT_DIG       11
#define HALF_MAX_10_EXP     +4
#define HALF_MAX_EXP        +16
#define HALF_MIN_10_EXP     -4
#define HALF_MIN_EXP        -13
#define HALF_RADIX          2
#define HALF_MAX            0x1.ffcp15h
#define HALF_MIN            0x1.0p-14h
#define HALF_EPSILON        0x1.0p-10h

// The following are missing... perhaps because NVIDIA's support
// for half-precision types is incomplete? Perhaps because it doesn't
// support those in OpenCL? Who knows.

// #define M_E_H
// #define M_LOG2E_H
// #define M_LOG10E_H
// #define M_LN2_H
// #define M_LN10_H
// #define M_PI_H
// #define M_PI_2_H
// #define M_PI_4_H
// #define M_1_PI_H
// #define M_2_PI_H
// #define M_2_SQRTPI_H
// #define M_SQRT2_H
// #define M_SQRT1_2_H


#if defined( __GNUC__ )
#define HUGE_VALF     __builtin_huge_valf()
   #define HUGE_VAL      __builtin_huge_val()
   #define NAN           __builtin_nanf( "" )
#else
#define HUGE_VALF     ((cl_float) 1e50)
#define HUGE_VAL      ((cl_double) 1e500)
#define NAN           nanf( "" )
#endif
#define MAXFLOAT         CL_FLT_MAX
#define INFINITY         CL_HUGE_VALF

// #endif // PORT_FROM_OPENCL_ENABLE_MATH_DEFINES

// 6.15.3.2. Integer Macros
// -------------------------

#define CHAR_BIT        8
#define CHAR_MAX        SCHAR_MAX
#define CHAR_MIN        SCHAR_MIN
#define INT_MAX         2147483647
#define INT_MIN         (-2147483647 - 1)
#define LONG_MAX        0x7fffffffffffffffL
#define LONG_MIN        (-0x7fffffffffffffffL - 1)
#define SCHAR_MAX       127
#define SCHAR_MIN       (-127 - 1)
#define SHRT_MAX        32767
#define SHRT_MIN        (-32767 - 1)
#define UCHAR_MAX       255
#define USHRT_MAX       65535
#define UINT_MAX        0xffffffff
#define ULONG_MAX       0xffffffffffffffffUL

#endif // __OPENCL_VERSION__
#endif // PORT_FROM_OPENCL_DEFINES_CUH_
