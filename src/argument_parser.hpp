/**
 * @file
 *
 * @brief Declaration and fallback implementation of opaque-type parsers,
 * for supporting the parsing of kernel-adapter-specific scalar kernel
 * arguments, passed as command-line arguments to the kernel runner.
 */
#ifndef GPU_KERNEL_RUNNER_PARSERS_HPP_
#define GPU_KERNEL_RUNNER_PARSERS_HPP_

#include "util/optional_and_any.hpp"

#include <string>
#include <sstream>

// namespace kernel_parameters {

using parser_type = any (*)(const std::string&);

template <typename T>
any parser(const std::string& str)
{
    static_assert(
        not std::is_pointer<T>::value and
        not std::is_reference<T>::value and
        not std::is_array<T>::value and
        not std::is_reference<T>::value and
        std::is_trivially_copyable<T>::value,
        "Invalid kernel parameter type");

    // Note: If we make this static, somehow the stringstream can
    // get itself into weird states
    std::istringstream iss;

    T result;
    iss.exceptions();
    iss.str(str);
    iss >> result;
    return result;
}

constexpr const parser_type no_parser = nullptr;

// } // namespace kernel_parameters

#endif // GPU_KERNEL_RUNNER_PARSERS_HPP_
