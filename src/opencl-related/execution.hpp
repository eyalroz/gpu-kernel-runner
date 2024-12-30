#ifndef KERNEL_RUNNER_OPENCL_EXECUTION_HPP_
#define KERNEL_RUNNER_OPENCL_EXECUTION_HPP_

#include "types.hpp"
#include "ugly_error_handling.hpp"

#include "../execution_context.hpp"

#include <spdlog/spdlog.h>

template<execution_ecosystem_t Ecosystem>
void validate_launch_configuration_(execution_context_t const& execution_context);

void set_opencl_kernel_arguments(cl::Kernel& kernel, marshalled_arguments_type& args);

// Note: This function assumes the stream has been clFinish'ed and
// all events have already occurred
durations_t compute_durations(std::vector<cl::Event> timing_events);

void launch_and_time_opencl_kernel(execution_context_t& execution_context, run_index_t run_index);

#endif // KERNEL_RUNNER_OPENCL_EXECUTION_HPP_
