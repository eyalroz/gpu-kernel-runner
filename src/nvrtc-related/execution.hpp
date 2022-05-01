#ifndef KERNEL_RUNNER_CUDA_EXECUTION_HPP_
#define KERNEL_RUNNER_CUDA_EXECUTION_HPP_

#include <common_types.hpp>
#include <nvrtc-related/ugly_error_handling.hpp>

void launch_time_and_sync_cuda_kernel(execution_context_t& context, run_index_t run_index)
{
    // TODO: This is ugly!
    cuda_api_call(cuCtxPushCurrent, context.cuda.context);
    void** alternative_format_arguments { nullptr };
    const auto& lc = context.kernel_launch_configuration.cuda;
    spdlog::info("Launching kernel {} (function name \"{}\")",
        context.options.kernel.key,
        context.options.kernel.function_name);

    CUstream stream_handle;
    cuda_api_call(cuStreamCreate, &stream_handle, CU_STREAM_NON_BLOCKING);
    spdlog::debug("Created a non-blocking stream.");
    // Sorry about this ugly code!
    struct {
        CUevent before, after;
    } events;
    if (context.options.time_with_events) {
        spdlog::debug("Creating events & recording \"before\" event.");
        cuda_api_call(cuEventCreate, &events.before, CU_EVENT_BLOCKING_SYNC);
        cuda_api_call(cuEventCreate, &events.after, CU_EVENT_BLOCKING_SYNC);
        cuda_api_call(cuEventRecord, events.before, stream_handle);
    }

    spdlog::debug("Passing {} arguments to the kernel.", context.finalized_arguments.pointers.size() - 1);
    cuda_api_call(
        cuLaunchKernel,
        context.cuda.built_kernel,
        lc.dimensions.grid.x,   lc.dimensions.grid.y, lc.dimensions.grid.z,
        lc.dimensions.block.x, lc.dimensions.block.y, lc.dimensions.block.z,
        lc.dynamic_shared_memory_size,
        stream_handle,
        const_cast<void**>(context.finalized_arguments.pointers.data()),
        alternative_format_arguments

    );
    if (context.options.time_with_events) {
        cuda_api_call(cuEventRecord, events.after, stream_handle);
    }
    cuda_api_call(cuStreamSynchronize, cuda::stream::default_stream_handle);
    cuda_api_call_noargs(cuCtxSynchronize);

    if (context.options.time_with_events) {
        float milliseconds_elapsed;
        cuda_api_call(cuEventElapsedTime, &milliseconds_elapsed, events.before, events.after);
        spdlog::info("Event-measured time of run {} of kernel {}: {:.0f} nsec",
            run_index+1, context.kernel_adapter_->kernel_function_name(), ((double) milliseconds_elapsed * 1000000.0));
    }
    cuda_api_call(cuCtxPopCurrent, nullptr);
}

#endif // KERNEL_RUNNER_CUDA_EXECUTION_HPP_
