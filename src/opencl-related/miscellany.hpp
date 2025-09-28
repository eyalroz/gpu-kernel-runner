#ifndef KERNEL_RUNNER_OPENCL_MISC_HPP_
#define KERNEL_RUNNER_OPENCL_MISC_HPP_

#include "types.hpp"

#include <iostream>
#include <string>
#include <cstring>
#include <iomanip>

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


template <>
inline void ensure_gpu_device_validity_<execution_ecosystem_t::opencl>(
    optional<unsigned> platform_id, int device_id, bool)
{
    constexpr const unsigned OpenCLDefaultPlatformID { 0 };
    auto actual_platform_id = platform_id.value_or(OpenCLDefaultPlatformID);
    spdlog::trace("Validating actual platform ID  {}", actual_platform_id);
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (actual_platform_id >= platforms.size()) {
        die ("Invalid platform index specified (outside the valid range 0.. {})", cuda::device::count()-1);
    }
    const auto& platform = platforms[actual_platform_id];
    cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties) (platform)(),
            0 };
    auto context = cl::Context{CL_DEVICE_TYPE_GPU, properties};
    auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
    if (device_id < 0 or device_id >= (int) devices.size())
        die ("Invalid device index specified (outside the valid range 0.. {})", cuda::device::count()-1);
    return;
}


#endif // KERNEL_RUNNER_OPENCL_MISC_HPP_
