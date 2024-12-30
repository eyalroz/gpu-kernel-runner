#ifndef KERNEL_RUNNER_CUDA_EXECUTION_HPP_
#define KERNEL_RUNNER_CUDA_EXECUTION_HPP_

#include "../execution_context.hpp"
#include "types.hpp"

template <execution_ecosystem_t ecosystem>
void validate_launch_configuration_(execution_context_t const& context);

template <>
void initialize_execution_context<execution_ecosystem_t::cuda>(execution_context_t& execution_context);

template<>
void validate_launch_configuration_<execution_ecosystem_t::cuda>(execution_context_t const& execution_context);

void launch_and_time_cuda_kernel(execution_context_t& execution_context, run_index_t run_index);

durations_t compute_durations(const std::vector<cuda_event_pair_t>& timing_events);

#endif // KERNEL_RUNNER_CUDA_EXECUTION_HPP_
