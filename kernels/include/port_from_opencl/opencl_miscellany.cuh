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
#ifndef PORT_FROM_OPENCL_MISCELLANY_CUH_
#define PORT_FROM_OPENCL_MISCELLANY_CUH_

#ifndef __OPENCL_VERSION__

// §6.7: Address space qualifiers
// ------------------------

#define __global
#define __private
// Note: _No_ definition of __local; it is not to be used directly
// Note: _No_ definition of __constant ... for now

// Note: _No_ definition of keywords without the double-underscore prefix, i.e. global, private, local, constant

// §6.8 Access qualifiers
// ------------------------

// _Not_ defining __read_only/read_only , __write_only/write_only,  __read_write/read_write , which
// apply to OpenCL images, that should not be used in kernels being ported from CUDA to OpenCL

// §6.9 Function qualifiers, §6.10. Storage-Class Specifiers
// ---------------------------------------------------------
#ifndef __kernel
#define __kernel extern "C" __global__
// TODO: Should we uncomment this next line?
// #define kernel __kernel
#endif

#ifndef restrict
#if defined(__CDT_PARSER__) || defined (__JETBRAINS_IDE__)
// The parsers are rather intolerant of restrict'ng C arrays.
#define restrict
#define __restrict
#else
#define restrict __restrict__
#define __restrict __restrict__
#endif
#endif // restrict
// and note __local is missing!

// §6.9.2. Optional Attribute Qualifiers
// -------------------------------------

// Have not defined: __attribute__((vec_type_hint(<type>)))
//
// TODO: Can we do something about  __attribute__((work_group_size_hint(X, Y, Z))) ?



#endif // __OPENCL_VERSION__
#endif // PORT_FROM_OPENCL_MISCELLANY_CUH_
