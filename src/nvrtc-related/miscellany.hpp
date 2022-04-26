#ifndef KERNEL_RUNNER_CUDA_MISC_HPP
#define KERNEL_RUNNER_CUDA_MISC_HPP

#include <common_types.hpp>
#include <spdlog/spdlog.h>

// This is a stub. It's not terribly hard to do, but it's more than a few lines
// of code - certainly if you want it to work on MacOs Windows as well as Linux.
optional<std::string> locate_cuda_include_directory()
{
    constexpr const char* cuda_root_env_var_name = "CUDA_PATH";
    optional<std::string> cuda_root = util::get_env(cuda_root_env_var_name);
    if (not cuda_root) {
        spdlog::warn("Environment variable {} is not set", cuda_root_env_var_name);
        return nullopt;
    }
    return filesystem::path(cuda_root.value()) / "include";
}

#endif // KERNEL_RUNNER_CUDA_MISC_HPP
