#ifndef KERNEL_RUNNER_CUDA_EXECUTION_HPP_
#define KERNEL_RUNNER_CUDA_EXECUTION_HPP_

#include <common_types.hpp>

template <>
void initialize_execution_context<execution_ecosystem_t::cuda>(execution_context_t& execution_context)
{
    auto device = cuda::device::get(execution_context.options.gpu_device_id);
    execution_context.cuda.context = device.create_context();
    execution_context.cuda.stream.emplace(execution_context.cuda.context->create_stream(cuda::stream::async));
    spdlog::trace("Created a CUDA context on GPU device {} ", execution_context.cuda.context->device_id());
}

void launch_time_and_sync_cuda_kernel(execution_context_t& execution_context, run_index_t run_index)
{
    auto& cuda_context = *execution_context.cuda.context;
    cuda::context::current::scoped_override_t cuda_context_for_this_scope(cuda_context);

    const auto& lc = execution_context.kernel_launch_configuration.cuda;
    spdlog::info("Launching kernel {} (function name \"{}\")",
                 execution_context.options.kernel.key,
                 execution_context.options.kernel.function_name);

    spdlog::debug("Created a non-blocking CUDA stream on device {}", cuda_context.device_id());
    struct event_pair_t {
        cuda::event_t before, after;
    } ;
    optional<event_pair_t> timing_events;
    if (execution_context.options.time_with_events) {
        spdlog::debug("Creating events & recording \"before\" event on the CUDA stream.");
        timing_events = {
            cuda_context.create_event(cuda::event::sync_by_blocking),
            cuda_context.create_event(cuda::event::sync_by_blocking)
        };
        execution_context.cuda.stream->enqueue.event(timing_events->before);
    }
    auto mangled_kernel_signature = execution_context.cuda.mangled_kernel_signature->c_str();
    auto kernel = execution_context.cuda.module->get_kernel(mangled_kernel_signature);

    spdlog::debug("Passing {} arguments to kernel {}",
        execution_context.finalized_arguments.pointers.size() - 1,
        execution_context.options.kernel.function_name.c_str());

    cuda::launch_type_erased(
        kernel,
       execution_context.cuda.stream.value(),
       lc,
       execution_context.finalized_arguments.pointers);

    if (execution_context.options.time_with_events) {
        execution_context.cuda.stream->enqueue.event(timing_events->after);
    }
    spdlog::debug("Launched run {} of kernel '{}'", run_index+1,
        execution_context.kernel_adapter_->kernel_function_name());

    execution_context.cuda.stream->synchronize();

    if (execution_context.options.time_with_events) {
        auto duration = cuda::event::time_elapsed_between(timing_events->before, timing_events->after);
        execution_context.durations.push_back(std::chrono::duration_cast<duration_t>(duration));
    }
}

#endif // KERNEL_RUNNER_CUDA_EXECUTION_HPP_
