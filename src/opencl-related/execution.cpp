
#include "execution.hpp"
#include "../util/spdlog-extra.hpp"

template <execution_ecosystem_t ecosystem>
void validate_launch_configuration_(execution_context_t const& context);

static void validate_opencl_launch_config_dims(execution_context_t const& execution_context)
{
    auto& dev = execution_context.opencl.device;
    auto dim_maxima  = dev.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
    (dim_maxima.size() == 3) or die(
        "Device {} reports an unsupported NDRange dimensionality: {}",
        execution_context.device_id, dim_maxima.size());
    size_t max_workgroup_size { 1 };
    for(int i = 0; i < 3; i++) {
        auto dimension = execution_context.kernel_launch_configuration.opencl.block_dimensions[i];
        (dimension == 0) or die("Launch configuration dimension {} specified as 0", i+1);
        (dimension >= dim_maxima[i]) or die(
            "Requested launch grid dimension of {} in axis {} is too large for OpenCL device {}",
            dimension, i+1, execution_context.device_id);
        max_workgroup_size *= dimension;
    }
    auto max_allowed_workgroup_size = dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    (max_workgroup_size <= max_allowed_workgroup_size) or die(
        "Requested launch grid with workgroups of size {}, exceeding the limit on device , ",
        max_workgroup_size, execution_context.device_id, max_allowed_workgroup_size);
}

template<>
void validate_launch_configuration_<execution_ecosystem_t::opencl>(execution_context_t const& execution_context)
{
    validate_opencl_launch_config_dims(execution_context);

    // Note: We don't validate:
    //
    // * Amount of dynamic shared memory
    // * Validity of dimensions for the specific kernel (as opposed to the device generally)
}

void set_opencl_kernel_arguments(
    cl::Kernel& kernel,
    marshalled_arguments_type& args)
{
    (args.pointers.size() == args.sizes.size()) or die(
        "Number of kernel arguments does not match number of kernel argument sizes");
    spdlog::debug("Passing {} arguments to the kernel...", args.pointers.size());
    for (cl_uint i = 0; i < args.pointers.size(); i++) {
        kernel.setArg(i, args.sizes[i], args.pointers[i]);
    }
    spdlog::debug("All arguments passed.");
}

static opencl_duration_type opencl_command_execution_time(const cl::Event& ev)
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

// Note: This function assumes the stream has been clFinish'ed and
// all events have already occurred
durations_t compute_durations(std::vector<cl::Event> timing_events)
{
    durations_t durations;
    for(const auto& event : timing_events) {
        auto duration = opencl_command_execution_time(event);
        durations.emplace_back(std::chrono::duration_cast<duration_t>(duration));
    }
    return durations;
    /*
    if (context.options.time_with_events) {
        kernel_execution_event_ptr->wait();
        auto time_elapsed = opencl_command_execution_time(*kernel_execution_event_ptr);
        context.durations.push_back(time_elapsed);
    }

     */
}

template <>
void initialize_execution_context<execution_ecosystem_t::opencl>(execution_context_t& execution_context)
{
    constexpr const unsigned OpenCLDefaultPlatformID { 0 };
    auto actual_platform_id = execution_context.options.platform_id.value_or(OpenCLDefaultPlatformID);
    spdlog::trace("actual platform ID is {}", actual_platform_id);
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms); // At this point we know our platform does exist
    const auto& platform = platforms[actual_platform_id];
    cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties) (platform)(),
            0 };
    execution_context.opencl.context = cl::Context{CL_DEVICE_TYPE_GPU, properties};
    auto devices = execution_context.opencl.context.getInfo<CL_CONTEXT_DEVICES>();
    // Device IDs happen to be ordinals into the devices array.
    execution_context.opencl.device = devices[(size_t) execution_context.options.gpu_device_id];
    constexpr const cl_command_queue_properties queue_properties { CL_QUEUE_PROFILING_ENABLE } ;
    execution_context.opencl.queue =
        cl::CommandQueue(execution_context.opencl.context, execution_context.opencl.device, queue_properties);
}

void launch_and_time_opencl_kernel(execution_context_t& execution_context, run_index_t run_index)
{
    auto lc = execution_context.kernel_launch_configuration;

    spdlog::info("Scheduling kernel run {1:>{0}}", util::naive_num_digits(execution_context.options.num_runs), run_index + 1);

    auto& events = execution_context.opencl.timing_events;
    if (execution_context.options.time_with_events) {
        events.emplace_back();
        // Notes:
        // 1. This will make the size equal run_index + 1;
        // 2. When creating an uninitialized event, no OpenCL API call is made
    }

    static const std::vector<cl::Event>* no_events_to_wait_on { nullptr };
    auto kernel_execution_event_ptr = [&]() -> cl::Event* {
        if (not execution_context.options.time_with_events) return nullptr;
        return &events[run_index];
    }();

    try {
        execution_context.opencl.queue.enqueueNDRangeKernel(
            execution_context.opencl.built_kernel,
            lc.opencl.offset(),
            lc.opencl.global_dims(),
            lc.opencl.local_dims(),
            no_events_to_wait_on,
            kernel_execution_event_ptr);
    }
    catch(cl::Error& e) {
        spdlog::error("Failed scheduling kernel: {}", clGetErrorString(e.err()) );
    }

    if (execution_context.options.sync_after_kernel_execution) {
        spdlog::debug("Waiting for run {1:>{0}} to conclude",
            util::naive_num_digits(execution_context.options.num_runs), run_index + 1);
        execution_context.opencl.queue.finish();
        spdlog::info("Kernel run {1:>{0}} concluded",
            util::naive_num_digits(execution_context.options.num_runs), run_index + 1);
    }
}
