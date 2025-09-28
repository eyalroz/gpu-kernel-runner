#ifndef KERNEL_RUNNER_CUDA_MISCELLANY_HPP_
#define KERNEL_RUNNER_CUDA_MISCELLANY_HPP_

#include "../common_types.hpp"
#include "../execution_context.hpp"

template <execution_ecosystem_t Ecosystem>
std::vector<filesystem::path> get_ecosystem_include_paths_();

template <>
std::vector<filesystem::path> get_ecosystem_include_paths_<execution_ecosystem_t::cuda>();

// This is a stub. It's not terribly hard to do, but it's more than a few lines
// of code - certainly if you want it to work on MacOs Windows as well as Linux.
optional<std::string> locate_cuda_include_directory();

size_t determine_required_shared_memory_size(execution_context_t const& context);
void enable_sufficient_shared_memory(execution_context_t const& context);
void enable_sufficient_shared_memory(execution_context_t const& context, size_t required_shmem_size);

#endif // KERNEL_RUNNER_CUDA_MISCELLANY_HPP_

