#ifndef KERNEL_RUNNER_OPENCL_MISC_HPP_
#define KERNEL_RUNNER_OPENCL_MISC_HPP_

#include "types.hpp"

#include <iostream>
#include <string>
#include <cstring>
#include <iomanip>

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
    return name;
}

inline bool uses_ptx(cl::Platform& platform)
{
    return (strcmp(get_name(platform).c_str(), "NVIDIA CUDA") == 0);
        // This weird comparison is due to observing that the
        // string returned by get_name may contain a trailing '\0'.
}

#endif // KERNEL_RUNNER_OPENCL_MISC_HPP_
