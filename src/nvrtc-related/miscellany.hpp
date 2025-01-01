#ifndef KERNEL_RUNNER_CUDA_MISCELLANY_HPP_
#define KERNEL_RUNNER_CUDA_MISCELLANY_HPP_

#include "../common_types.hpp"

template <execution_ecosystem_t Ecosystem>
std::vector<filesystem::path> get_ecosystem_include_paths_();

template <>
inline std::vector<filesystem::path> get_ecosystem_include_paths_<execution_ecosystem_t::cuda>()
{
    std::vector<filesystem::path> result;
    auto cuda_include_dir = locate_cuda_include_directory();
    if (cuda_include_dir) {
        spdlog::debug("Using CUDA include directory {}", cuda_include_dir.value());
        result.emplace_back(cuda_include_dir.value());
    }
    else {
        spdlog::warn("Cannot locate CUDA include directory - trying to build the kernel with it missing.");
    }
    return result;
}

#endif // KERNEL_RUNNER_CUDA_MISCELLANY_HPP_

