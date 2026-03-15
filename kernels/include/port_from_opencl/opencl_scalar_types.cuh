/**
 * @file
 *
 * @brief Definitions for using scalar types from originally-OpenCL-C code in 
 * CUDA code. This is part of a set of files which, included together, allow
 * for using slightly-tweaked OpenCL C kernel code as CUDA code. 
 *
 * @copyright (c) 2020-2026, GE HealthCare
 * @copyright (c) 2020-2026, Eyal Rozenberg
 *
 * @license BSD 3-clause license; see the `LICENSE` file or
 * @url https://opensource.org/licenses/BSD-3-Clause
 *
 * @note Covered OpenCL C language spec section: 6.3.1. (Built-in Scalar Data Types),
 * but without any builtin functions which take or return scalar types defined here.
 *
 */
#ifndef PORT_FROM_OPENCL_SCALAR_TYPES_CUH_
#define PORT_FROM_OPENCL_SCALAR_TYPES_CUH_

#ifndef __OPENCL_VERSION__

#if !defined(__CDT_PARSER__) && !defined (__JETBRAINS_IDE__)
// Parsers may fail to recognize a reasonable default-C++-headers path for kernel files
#include <cstdint>
#include <cstddef> // for size_t
#include <climits>
#endif

#if __cplusplus < 201103L
#error "This file requires compiling using C++11 or later"
#endif

#ifdef PORT_FROM_OPENCL_ENABLE_HALF_PRECISION
#include <cuda_fp16.h>
#endif

using size_t = unsigned long;
#ifndef UNSIGNED_INTEGRAL_SHORTHANDS_DEFINED
typedef unsigned char  uchar;
typedef unsigned short ushort;
typedef unsigned int   uint;
typedef unsigned long  ulong;
#endif // UNSIGNED_INTEGRAL_SHORTHANDS_DEFINED
#ifndef SIGNED_INTEGRAL_SHORTHANDS_DEFINED
typedef signed char  schar;
typedef signed short sshort;
typedef signed int   sint;
typedef signed long  slong;
#endif // SIGNED_INTEGRAL_SHORTHANDS_DEFINED

#if !defined(__CDT_PARSER__) && !defined (__JETBRAINS_IDE__)
// The IDEs (well, at least JetBrians) seem to already know about these types in
// the global namespace. Plus, we've not included the relevant C-standard-library
// headerds for the following to be defined.
using std::ptrdiff_t;
using std::intptr_t;
using std::uintptr_t;
#endif

#endif // __OPENCL_VERSION__
#endif // PORT_FROM_OPENCL_SCALAR_TYPES_CUH_
