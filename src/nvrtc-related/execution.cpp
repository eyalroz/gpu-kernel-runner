
#include "execution.hpp"

#include "../util/spdlog-extra.hpp"

template <execution_ecosystem_t ecosystem>
void validate_launch_configuration_(execution_context_t const& context);

template <>
void initialize_execution_context<execution_ecosystem_t::cuda>(execution_context_t& execution_context)
{
    auto device = cuda::device::get(execution_context.options.gpu_device_id);
    execution_context.cuda.context = device.create_context();
    execution_context.cuda.stream.emplace(execution_context.cuda.context->create_stream(cuda::stream::async));
    spdlog::trace("Created a CUDA context on GPU device {} ", execution_context.cuda.context->device_id());
}

template<>
void validate_launch_configuration_<execution_ecosystem_t::cuda>(execution_context_t const& execution_context)
{
    auto const& kernel = execution_context.cuda.kernel.value();
    auto launch_config = execution_context.kernel_launch_configuration.cuda;
    cuda::detail_::validate(launch_config);
    cuda::detail_::validate_block_dimension_compatibility(kernel, launch_config.dimensions.block);
    cuda::detail_::validate_compatibility(kernel.device(), launch_config);
    // TODO: This should really go in the API wrappers validation code
    cuda::memory::shared::size_t max_dynamic_shmem_size = kernel.get_attribute(cuda::kernel::attribute_t::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES);
    (launch_config.dynamic_shared_memory_size <= max_dynamic_shmem_size) or die(
        "The requested launch configuration would have {} bytes of dynamic shared memory, but "
        "the kernel only supports up to {} bytes on device {}", launch_config.dynamic_shared_memory_size,
        max_dynamic_shmem_size, execution_context.device_id);
}

void launch_and_time_cuda_kernel(execution_context_t& execution_context, run_index_t run_index)
{
    auto& cuda_context = *execution_context.cuda.context;
    cuda::context::current::scoped_override_t cuda_context_for_this_scope(cuda_context);

    const auto& lc = execution_context.kernel_launch_configuration.cuda;

    auto& events = execution_context.cuda.timing_events;

    if (execution_context.options.time_with_events) {
        events.push_back({
            cuda_context.create_event(cuda::event::sync_by_blocking),
            cuda_context.create_event(cuda::event::sync_by_blocking)
            });
        execution_context.cuda.stream->enqueue.event(events[events.size() - 1].before);
    }
    spdlog::info("Scheduling kernel run {1:>{0}}",
        util::naive_num_digits(execution_context.options.num_runs), run_index + 1);

    cuda::launch_type_erased(
        execution_context.cuda.kernel.value(),
        execution_context.cuda.stream.value(),
        lc,
        execution_context.finalized_arguments.pointers);

    if (execution_context.options.time_with_events) {
        execution_context.cuda.stream->enqueue.event(events[events.size() - 1].after);
    }

    if (execution_context.options.sync_after_kernel_execution) {
        spdlog::debug("Waiting for run {1:>{0}} to conclude",
            util::naive_num_digits(execution_context.options.num_runs), run_index + 1);
        execution_context.cuda.stream->synchronize();
    }
}

durations_t compute_durations(const std::vector<cuda_event_pair_t>& timing_events)
{
    durations_t durations;
    for(const auto& pair : timing_events) {
        auto duration = cuda::event::time_elapsed_between(pair.before, pair.after);
        durations.emplace_back(std::chrono::duration_cast<duration_t>(duration));
    }
    return durations;
}
