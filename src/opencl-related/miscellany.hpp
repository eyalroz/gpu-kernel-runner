#ifndef KERNEL_RUNNER_OPENCL_MISC_HPP_
#define KERNEL_RUNNER_OPENCL_MISC_HPP_

#include "types.hpp"
#include "../common_types.hpp"

#include <iostream>
#include <string>
#include <cstring>
#include <iomanip>
#include <spdlog/common.h>

template <execution_ecosystem_t Ecosystem>
std::vector<filesystem::path> get_ecosystem_include_paths_();

template <>
inline std::vector<filesystem::path> get_ecosystem_include_paths_<execution_ecosystem_t::opencl>()
{
    return {};
}


inline std::ostream& operator<<(std::ostream& os, const cl::NDRange& rng)
{
    auto sizes = static_cast<const size_t*>(rng);
    os << '(';
    switch (rng.dimensions()) {
    case 1: os << sizes[0]; break;
    case 2: os << sizes[0] << ", " << sizes[1]; break;
    case 3: os << sizes[0] << ", " << sizes[1] << ", " << sizes[2]; break;
    }
    os << ')' << std::endl;;
    return os;
}

// Note that for NVIDIA CUDA, the platform name storage space
// may be larger
inline std::string get_name(cl::Platform& platform) {
    std::string name;
    platform.getInfo(CL_PLATFORM_NAME, &name);
    // The OpenCL API may return the string with an embedded null character. Annoying!
    auto nul_char_pos = name.find_first_of('\0');
    if (nul_char_pos != std::string::npos) {
        name.resize(nul_char_pos);
    }
    return name;
}

inline bool uses_ptx(cl::Platform& platform)
{
    return (strcmp(get_name(platform).c_str(), "NVIDIA CUDA") == 0);
        // This weird comparison is due to observing that the
        // string returned by get_name may contain a trailing '\0'.
}

template <execution_ecosystem_t Ecosystem>
void ensure_gpu_device_validity_(
    optional<unsigned>     platform_id,
    int                    device_id,
    bool                   need_ptx);

template <>
void ensure_gpu_device_validity_<execution_ecosystem_t::opencl>(
    optional<unsigned> platform_id, int device_id, bool);

cl_channel_order get_image_channel_order(device_side_buffer_info_t const & buffer_info);
cl_channel_type get_image_channel_type(device_side_buffer_info_t const & buffer_info);

std::string format_as(const cl::size_t<3>& dimensions);

// template <>
// class spdlog::fmt_lib::formatter<dimensions_t> {
// public:
//     constexpr auto parse (format_parse_context& ctx) { return ctx.begin(); }
//     template <typename Context>
//     constexpr auto format (dimensions_t const& dimensions, Context& ctx)
//     {
//         if (dimensions[1] == 0) { return format_to(ctx.out(), "({})", dimensions[0]); }
//         if (dimensions[2] == 0) { return format_to(ctx.out(), "({},{})", dimensions[0], dimensions[1]); }
//         return format_to(ctx.out(), "({},{},{})", dimensions[0], dimensions[1], dimensions[2]);
//     }
// };

#endif // KERNEL_RUNNER_OPENCL_MISC_HPP_
