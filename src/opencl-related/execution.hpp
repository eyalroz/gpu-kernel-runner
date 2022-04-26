#ifndef KERNEL_RUNNER_OPENCL_EXECUTION_HPP_
#define KERNEL_RUNNER_OPENCL_EXECUTION_HPP_

#include <common_types.hpp>
#include <opencl-related/types.hpp>
#include <opencl-related/ugly_error_handling.hpp>

void set_opencl_kernel_arguments(
    cl::Kernel& kernel,
    marshalled_arguments_type& args)
{
    if (args.pointers.size() != args.sizes.size()) {
        throw std::runtime_error("Number of kernel arguments does not match number of kernel argument sizes");
    }
    spdlog::debug("Passing {} arguments to the kernel...", args.pointers.size());
    for (auto i = 0; i < args.pointers.size(); i++) {
        kernel.setArg(i, args.sizes[i], args.pointers[i]);
    }
    spdlog::debug("All arguments passed.");
}

opencl_duration_type opencl_command_execution_time(cl::Event& ev)
{
    cl_ulong t_start { 0 }, t_end { 0 };
    try {
        ev.getProfilingInfo(CL_PROFILING_COMMAND_START, &t_start);
        ev.getProfilingInfo(CL_PROFILING_COMMAND_END, &t_end);
    }
    catch(cl::Error& e) {
        spdlog::error("Failed obtaining execution event start or end time (using {}): {}", e.what(), clGetErrorString(e.err()) );
    }
    return opencl_duration_type{t_end - t_start};
}


void launch_time_and_sync_opencl_kernel(execution_context_t& context, run_index_t run_index)
{
    auto lc = context.kernel_launch_configuration;
    cl::Event kernel_execution; // When uninitialized, no OpenCL API call is made

    set_opencl_kernel_arguments(context.opencl.built_kernel, context.finalized_arguments);

    const std::vector<cl::Event>* no_events_to_wait_on { nullptr };
    auto kernel_execution_event_ptr =
        context.parsed_inspecific_options.time_with_events ? &kernel_execution : nullptr;

    try {
        context.opencl.queue.enqueueNDRangeKernel(
            context.opencl.built_kernel,
            lc.opencl.offset(),
            lc.opencl.global_dims(),
            lc.opencl.local_dims(),
            no_events_to_wait_on,
            kernel_execution_event_ptr);
    }
    catch(cl::Error& e) {
        spdlog::error("Failed enqueuing kernel: {}", clGetErrorString(e.err()) );
    }

    spdlog::debug("Launched run {} of kernel '{}'", run_index+1, context.kernel_adapter_->kernel_function_name());

    if (context.parsed_inspecific_options.time_with_events) {
        kernel_execution.wait();
        auto time_elapsed = opencl_command_execution_time(kernel_execution);
        spdlog::info("Event-measured time of run {} of kernel {}: {} nsec",
            run_index+1, context.kernel_adapter_->kernel_function_name(), time_elapsed.count());
    }
    else {
        context.opencl.queue.finish(); // To make sure we catch any possible errors here.
    }
}

#endif // KERNEL_RUNNER_OPENCL_EXECUTION_HPP_