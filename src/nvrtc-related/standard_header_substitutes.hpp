// The code in this file was extracted and adapted from the
// NVIDIA jitify library sources, at:
// https://github.com/NVIDIA/jitify/
//
// And is made available under the terms of the BSD 3-Clause license:
// https://github.com/NVIDIA/jitify/blob/master/LICENSE

#ifndef KERNEL_RUNNER_STANDARD_HEADER_SUBSTITUTES_HPP_
#define KERNEL_RUNNER_STANDARD_HEADER_SUBSTITUTES_HPP_

#include <unordered_map>
#include <utility>
#include <vector>

static const char* jitsafe_header_stdint_h =
    "#pragma once\n"
    "#ifndef JITSAFE_HEADER_STDINT_H_\n"
    "#define JITSAFE_HEADER_STDINT_H_\n"
    "#include <climits>\n"
    "namespace __jitify_stdint_ns {\n"
    "typedef signed char      int8_t;\n"
    "typedef signed short     int16_t;\n"
    "typedef signed int       int32_t;\n"
    "typedef signed long long int64_t;\n"
    "typedef signed char      int_fast8_t;\n"
    "typedef signed short     int_fast16_t;\n"
    "typedef signed int       int_fast32_t;\n"
    "typedef signed long long int_fast64_t;\n"
    "typedef signed char      int_least8_t;\n"
    "typedef signed short     int_least16_t;\n"
    "typedef signed int       int_least32_t;\n"
    "typedef signed long long int_least64_t;\n"
    "typedef signed long long intmax_t;\n"
    "typedef signed long      intptr_t; //optional\n"
    "typedef unsigned char      uint8_t;\n"
    "typedef unsigned short     uint16_t;\n"
    "typedef unsigned int       uint32_t;\n"
    "typedef unsigned long long uint64_t;\n"
    "typedef unsigned char      uint_fast8_t;\n"
    "typedef unsigned short     uint_fast16_t;\n"
    "typedef unsigned int       uint_fast32_t;\n"
    "typedef unsigned long long uint_fast64_t;\n"
    "typedef unsigned char      uint_least8_t;\n"
    "typedef unsigned short     uint_least16_t;\n"
    "typedef unsigned int       uint_least32_t;\n"
    "typedef unsigned long long uint_least64_t;\n"
    "typedef unsigned long long uintmax_t;\n"
    "#define INT8_MIN    SCHAR_MIN\n"
    "#define INT16_MIN   SHRT_MIN\n"
    "#if defined _WIN32 || defined _WIN64\n"
    "#define WCHAR_MIN   SHRT_MIN\n"
    "#define WCHAR_MAX   SHRT_MAX\n"
    "typedef unsigned long long uintptr_t; //optional\n"
    "#else\n"
    "#define WCHAR_MIN   INT_MIN\n"
    "#define WCHAR_MAX   INT_MAX\n"
    "typedef unsigned long      uintptr_t; //optional\n"
    "#endif\n"
    "#define INT32_MIN   INT_MIN\n"
    "#define INT64_MIN   LLONG_MIN\n"
    "#define INT8_MAX    SCHAR_MAX\n"
    "#define INT16_MAX   SHRT_MAX\n"
    "#define INT32_MAX   INT_MAX\n"
    "#define INT64_MAX   LLONG_MAX\n"
    "#define UINT8_MAX   UCHAR_MAX\n"
    "#define UINT16_MAX  USHRT_MAX\n"
    "#define UINT32_MAX  UINT_MAX\n"
    "#define UINT64_MAX  ULLONG_MAX\n"
    "#define INTPTR_MIN  LONG_MIN\n"
    "#define INTMAX_MIN  LLONG_MIN\n"
    "#define INTPTR_MAX  LONG_MAX\n"
    "#define INTMAX_MAX  LLONG_MAX\n"
    "#define UINTPTR_MAX ULONG_MAX\n"
    "#define UINTMAX_MAX ULLONG_MAX\n"
    "#define PTRDIFF_MIN INTPTR_MIN\n"
    "#define PTRDIFF_MAX INTPTR_MAX\n"
    "#define SIZE_MAX    UINT64_MAX\n"
    "} // namespace __jitify_stdint_ns\n"
    "namespace std { using namespace __jitify_stdint_ns; }\n"
    "using namespace __jitify_stdint_ns;\n"
    "#endif // JITSAFE_HEADER_STDINT_H_\n"
    ;

static const char* jitsafe_header_limits_h = R"(
#pragma once
#ifndef JITSAFE_HEADER_LIMITS_H_
#define JITSAFE_HEADER_LIMITS_H_
#if defined _WIN32 || defined _WIN64
 #define __WORDSIZE 32
#else
 #if defined __x86_64__ && !defined __ILP32__
  #define __WORDSIZE 64
 #else
  #define __WORDSIZE 32
 #endif
#endif
#define MB_LEN_MAX  16
#define CHAR_BIT    8
#define SCHAR_MIN   (-128)
#define SCHAR_MAX   127
#define UCHAR_MAX   255
enum {
  _JITIFY_CHAR_IS_UNSIGNED = (char)-1 >= 0,
  CHAR_MIN = _JITIFY_CHAR_IS_UNSIGNED ? 0 : SCHAR_MIN,
  CHAR_MAX = _JITIFY_CHAR_IS_UNSIGNED ? UCHAR_MAX : SCHAR_MAX,
};
#define SHRT_MIN    (-32768)
#define SHRT_MAX    32767
#define USHRT_MAX   65535
#define INT_MIN     (-INT_MAX - 1)
#define INT_MAX     2147483647
#define UINT_MAX    4294967295U
#if __WORDSIZE == 64
 # define LONG_MAX  9223372036854775807L
#else
 # define LONG_MAX  2147483647L
#endif
#define LONG_MIN    (-LONG_MAX - 1L)
#if __WORDSIZE == 64
 #define ULONG_MAX  18446744073709551615UL
#else
 #define ULONG_MAX  4294967295UL
#endif
#define LLONG_MAX  9223372036854775807LL
#define LLONG_MIN  (-LLONG_MAX - 1LL)
#define ULLONG_MAX 18446744073709551615ULL
#endif // JITSAFE_HEADER_LIMITS_H_
)";

static const char* jitsafe_header_stddef_h =
    "#pragma once\n"
    "#ifndef JITSAFE_HEADER_STDDEF_H_\n"
    "#define JITSAFE_HEADER_STDDEF_H_\n"
    "#include <climits>\n"
    "namespace __jitify_stddef_ns {\n"
    "#if __cplusplus >= 201103L\n"
    "typedef decltype(nullptr) nullptr_t;\n"
    "#if defined(_MSC_VER)\n"
    "  typedef double max_align_t;\n"
    "#elif defined(__APPLE__)\n"
    "  typedef long double max_align_t;\n"
    "#else\n"
    "  // Define max_align_t to match the GCC definition.\n"
    "  typedef struct {\n"
    "    long long __jitify_max_align_nonce1\n"
    "        __attribute__((__aligned__(__alignof__(long long))));\n"
    "    long double __jitify_max_align_nonce2\n"
    "        __attribute__((__aligned__(__alignof__(long double))));\n"
    "  } max_align_t;\n"
    "#endif\n"
    "#endif  // __cplusplus >= 201103L\n"
    "#if __cplusplus >= 201703L\n"
    "enum class byte : unsigned char {};\n"
    "#endif  // __cplusplus >= 201703L\n"
    "} // namespace __jitify_stddef_ns\n"
    "namespace std {\n"
    "  // NVRTC provides built-in definitions of ::size_t and ::ptrdiff_t.\n"
    "  using ::size_t;\n"
    "  using ::ptrdiff_t;\n"
    "  using namespace __jitify_stddef_ns;\n"
    "} // namespace std\n"
    "using namespace __jitify_stddef_ns;\n"
    "#endif // JITSAFE_HEADER_STDDEF_H_\n"
    ;

static const char* jitsafe_header_stdio_h =
    "#pragma once\n"
    "#ifndef JITSAFE_HEADER_STDIO_H_\n"
    "#define JITSAFE_HEADER_STDIO_H_\n"
    "#include <stddef.h>\n"
    "#define FILE int\n"
    "int fflush ( FILE * stream );\n"
    "int printf(const char *format, ...);\n"
    "int fprintf ( FILE * stream, const char * format, ... );\n"
    "#endif // JITSAFE_HEADER_STDIO_H_\n"
    ;


// TODO: This is incomplete (missing binary and integer funcs, macros,
// constants, types)
static const char* jitsafe_header_math_h =
    "#pragma once\n"
    "#ifndef JITSAFE_HEADER_MATH_H_\n"
    "#define JITSAFE_HEADER_MATH_H_\n"
    "namespace __jitify_math_ns {\n"
    "#if __cplusplus >= 201103L\n"
    "#define DEFINE_MATH_UNARY_FUNC_WRAPPER(f) \\\n"
    "   inline double      f(double x)         { return ::f(x); } \\\n"
    "   inline float       f##f(float x)       { return ::f(x); } \\\n"
    "   /*inline long double f##l(long double x) { return ::f(x); }*/ \\\n"
    "   inline float       f(float x)          { return ::f(x); } \\\n"
    "   /*inline long double f(long double x)    { return ::f(x); }*/\n"
    "#else\n"
    "#define DEFINE_MATH_UNARY_FUNC_WRAPPER(f) \\\n"
    "   inline double      f(double x)         { return ::f(x); } \\\n"
    "   inline float       f##f(float x)       { return ::f(x); } \\\n"
    "   /*inline long double f##l(long double x) { return ::f(x); }*/\n"
    "#endif\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(cos)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(sin)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(tan)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(acos)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(asin)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(atan)\n"
    "template<typename T> inline T atan2(T y, T x) { return ::atan2(y, x); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(cosh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(sinh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(tanh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(exp)\n"
    "template<typename T> inline T frexp(T x, int* exp) { return ::frexp(x, "
    "exp); }\n"
    "template<typename T> inline T ldexp(T x, int  exp) { return ::ldexp(x, "
    "exp); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(log)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(log10)\n"
    "template<typename T> inline T modf(T x, T* intpart) { return ::modf(x, "
    "intpart); }\n"
    "template<typename T> inline T pow(T x, T y) { return ::pow(x, y); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(sqrt)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(ceil)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(floor)\n"
    "template<typename T> inline T fmod(T n, T d) { return ::fmod(n, d); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(fabs)\n"
    "template<typename T> inline T abs(T x) { return ::abs(x); }\n"
    "#if __cplusplus >= 201103L\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(acosh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(asinh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(atanh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(exp2)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(expm1)\n"
    "template<typename T> inline int ilogb(T x) { return ::ilogb(x); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(log1p)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(log2)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(logb)\n"
    "template<typename T> inline T scalbn (T x, int n)  { return ::scalbn(x, "
    "n); }\n"
    "template<typename T> inline T scalbln(T x, long n) { return ::scalbn(x, "
    "n); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(cbrt)\n"
    "template<typename T> inline T hypot(T x, T y) { return ::hypot(x, y); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(erf)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(erfc)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(tgamma)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(lgamma)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(trunc)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(round)\n"
    "template<typename T> inline long lround(T x) { return ::lround(x); }\n"
    "template<typename T> inline long long llround(T x) { return ::llround(x); "
    "}\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(rint)\n"
    "template<typename T> inline long lrint(T x) { return ::lrint(x); }\n"
    "template<typename T> inline long long llrint(T x) { return ::llrint(x); "
    "}\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(nearbyint)\n"
    // TODO: remainder, remquo, copysign, nan, nextafter, nexttoward, fdim,
    // fmax, fmin, fma
    "#endif\n"
    "#undef DEFINE_MATH_UNARY_FUNC_WRAPPER\n"
    "} // namespace __jitify_math_ns\n"
    "namespace std { using namespace __jitify_math_ns; }\n"
    "#define M_PI 3.14159265358979323846\n"
    // Note: Global namespace already includes CUDA math funcs
    "//using namespace __jitify_math_ns;\n"
    "#endif // JITSAFE_HEADER_MATH_H_\n"
    ;


static const char* kernel_runner_preinclude_h =
    "#pragma once\n"
    "#ifndef KERNEL_RUNNER_PREINCLUDE_H_\n"
    "#define KERNEL_RUNNER_PREINCLUDE_H_\n"
    "#define __UINT_LEAST8__      unsigned char\n"
    "#define __SIG_ATOMIC__       int\n"
    "#define __UINTMAX__          long unsigned int\n"
    "#define __INT_FAST16__       long int\n"
    "#define __INT_FAST64__       long int\n"
    "#define __UINT8__            unsigned char\n"
    "#define __INT_FAST32__       long int\n"
    "#define __UINT_LEAST16__     short unsigned int\n"
    "#define __SIZE__             long unsigned int\n"
    "#define __INT8__             signed char\n"
    "#define __INT_LEAST16__      short int\n"
    "#define __UINT_LEAST64__     long unsigned int\n"
    "#define __UINT_FAST16__      long unsigned int\n"
    "#define __CHAR16__           short unsigned int\n"
    "#define __INT_LEAST64__      long int\n"
    "#define __INT16__            short int\n"
    "#define __INT_LEAST8__       signed char\n"
    "#define __INTPTR__           long int\n"
    "#define __UINT16__           short unsigned int\n"
    "#define __WCHAR__            int\n"
    "#define __UINT_FAST64__      long unsigned int\n"
    "#define __INT64__            long int\n"
    "#define __WINT__             unsigned int\n"
    "#define __UINT_LEAST32__     unsigned int\n"
    "#define __INT_LEAST32__      int\n"
    "#define __INT_FAST8__        signed char\n"
    "#define __UINT64__           long unsigned int\n"
    "#define __UINT_FAST32__      long unsigned int\n"
    "#define __CHAR32__           unsigned int\n"
    "#define __INT32__            int\n"
    "#define __INTMAX__           long int\n"
    "#define __PTRDIFF__          long int\n"
    "#define __UINT32__           unsigned int\n"
    "#define __UINTPTR__          long unsigned int\n"
    "#define __UINT_FAST8__       unsigned char\n"
    "\n"
    "#define _GLIBCXX_USE_CXX11_ABI 0\n"
    "#endif // KERNEL_RUNNER_PREINCLUDE_H_\n";

/*
// WAR: These need to be pre-included as a workaround for NVRTC implicitly using
// /usr/include as an include path. The other built-in headers will be included
// lazily as needed.
static const char* preinclude_jitsafe_header_names[] = {
    "limits.h",
    "stddef.h",
    "stdio.h",
    "kernel_runner_preinclude.h"
};
*/

static auto& get_zipped_jitsafe_headers() {
  static const std::vector<std::pair<const char*, const char*>> jitsafe_headers_map = {
      {"kernel_runner_preinclude.h", kernel_runner_preinclude_h},
//      {"jitify_preinclude.h", jitsafe_header_preinclude_h},
//      {"float.h", jitsafe_header_float_h},
//      {"cfloat", jitsafe_header_float_h},
      {"limits.h", jitsafe_header_limits_h},
      {"climits", jitsafe_header_limits_h},
      {"stdint.h", jitsafe_header_stdint_h},
      {"cstdint", jitsafe_header_stdint_h},
      {"stddef.h", jitsafe_header_stddef_h},
      {"cstddef", jitsafe_header_stddef_h},
//      {"stdlib.h", jitsafe_header_stdlib_h},
//      {"cstdlib", jitsafe_header_stdlib_h},
      {"stdio.h", jitsafe_header_stdio_h},
      {"cstdio", jitsafe_header_stdio_h},
//      {"string.h", jitsafe_header_string_h},
//      {"cstring", jitsafe_header_cstring},
//      {"iterator", jitsafe_header_iterator},
//      {"limits", jitsafe_header_limits},
//      {"type_traits", jitsafe_header_type_traits},
//      {"utility", jitsafe_header_utility},
      {"math.h", jitsafe_header_math_h},
      {"cmath", jitsafe_header_math_h},
//      {"memory.h", jitsafe_header_memory_h},
//      {"complex", jitsafe_header_complex},
//      {"iostream", jitsafe_header_iostream},
//      {"ostream", jitsafe_header_ostream},
//      {"istream", jitsafe_header_istream},
//      {"sstream", jitsafe_header_sstream},
//      {"vector", jitsafe_header_vector},
//      {"string", jitsafe_header_string},
//      {"stdexcept", jitsafe_header_stdexcept},
//      {"mutex", jitsafe_header_mutex},
//      {"algorithm", jitsafe_header_algorithm},
//      {"time.h", jitsafe_header_time_h},
//      {"ctime", jitsafe_header_time_h},
  };
  return jitsafe_headers_map;
}

auto get_standard_header_substitutes() {
    // This is a sort of an unzip function - vector-of-pairs into pair-of-vectors
    const auto& zipped = get_zipped_jitsafe_headers();
    std::vector<const char*> header_names;
    std::vector<const char*> header_sources;
    header_names.reserve(zipped.size());
    header_sources.reserve(zipped.size());
    for(const auto& pair : zipped) {
        header_names.push_back(pair.first);
        header_sources.push_back(pair.second);
    }
    return std::make_pair(header_names, header_sources);
}

#endif // KERNEL_RUNNER_STANDARD_HEADER_SUBSTITUTES_HPP_