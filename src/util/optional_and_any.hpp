#ifndef UTIL_OPTIONAL_AND_ANY_HPP_
#define UTIL_OPTIONAL_AND_ANY_HPP_

#if __cplusplus >= 201703L
#include <optional>
#include <any>
using std::optional;
using std::nullopt;
using std::any;
using std::any_cast;
#else
static_assert(__cplusplus >= 201402L, "C++2014 is required to compile this program");
#include <experimental/optional>
#include <experimental/any>
using std::experimental::optional;
using std::experimental::nullopt;
using std::experimental::any;
using std::experimental::any_cast;
#endif

#endif // UTIL_OPTIONAL_AND_ANY_HPP_
