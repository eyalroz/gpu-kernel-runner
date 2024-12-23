#include "buffer_ops.hpp"

#include "kernel_adapter.hpp"
#include "util/buffer_io.hpp"

#include <spdlog/spdlog.h>

using std::size_t;
using std::string;

host_buffers_t read_input_buffers_from_files(
    const name_set&           buffer_names,
    const maybe_string_map&   filenames,
    const filesystem::path&   buffer_directory)
{
    spdlog::debug("Reading input buffers from files.");

    host_buffers_t result;
    std::unordered_map<string, filesystem::path> buffer_paths;
    for(const auto& name : buffer_names) {
        // Note: Even though the map is of optional<string>, we expect all relevant buffers
        // to have already had names determined
        auto path = maybe_prepend_base_dir(buffer_directory, filenames.at(name).value());
        try {
            spdlog::debug("Reading buffer '{}' from {}", name, path.native());
            host_buffer_t buffer = util::read_input_file(path);
            spdlog::debug("Have read buffer '{}': {} bytes from {}", name, buffer.size(), path.native());
            result.emplace(name, std::move(buffer));
        }
        catch(std::exception& ex) {
            spdlog::critical("Failed reading buffer '{}' from {}: {}", name, path.native(), ex.what());
            throw;
        }
    }
    return result;
}

void read_input_buffers_from_files(execution_context_t& context)
{
    auto input_buffer_names = buffer_names(*context.kernel_adapter_,
        [&](const auto& spd) {
            return is_input(spd.direction) and spd.kind == kernel_parameters::kind_t::buffer;
        }
    );
    context.buffers.host_side.inputs = read_input_buffers_from_files(
        input_buffer_names,
        context.buffers.filenames.inputs,
        context.options.buffer_base_paths.input);
}

// This function is synchronous and not effect by the sync_after_buffer_op option
void copy_buffer_to_device(
    const execution_context_t& context,
    const std::string&         buffer_name,
    const device_buffer_type&  device_side_buffer,
    const host_buffer_t&       host_side_buffer)
{
    if (context.ecosystem == execution_ecosystem_t::cuda) {
        spdlog::debug("Copying buffer '{}' (size {} bytes): host-side {} -> device-side {}",
            buffer_name, host_side_buffer.size(), (void *) host_side_buffer.data(),
            (void *) device_side_buffer.cuda.data());
        cuda::context::current::scoped_override_t scoped_context_override{ *context.cuda.context };
        cuda::memory::copy(device_side_buffer.cuda.data(), host_side_buffer.data(), host_side_buffer.size());
    } else { // OpenCL
        const constexpr auto blocking { CL_TRUE };
        context.opencl.queue.enqueueWriteBuffer(device_side_buffer.opencl, blocking, 0, host_side_buffer.size(),
            host_side_buffer.data());
    }
}

void copy_buffer_on_device(
    execution_ecosystem_t      ecosystem,
    cl::CommandQueue*          queue,
    const device_buffer_type&  destination,
    const device_buffer_type&  origin)
{
    if (ecosystem == execution_ecosystem_t::cuda) {
        cuda::memory::copy(destination.cuda.data(), origin.cuda.data(), destination.cuda.size());
    } else { // OpenCL
        size_t size;
        origin.opencl.getInfo(CL_MEM_SIZE, &size);
        queue->enqueueCopyBuffer(origin.opencl, destination.opencl, 0, 0, size);
    }
}

void gpu_sync(const execution_context_t& context) {
    if (context.ecosystem == execution_ecosystem_t::cuda) {
        context.cuda.context->synchronize();
    }
    else {
        context.opencl.queue.finish();
    }
}

void copy_input_buffers_to_device(const execution_context_t& context)
{
    spdlog::debug("Copying input (non-inout) buffers to the GPU.");
    for(const auto& input_pair : context.buffers.host_side.inputs) {
        const auto& name = input_pair.first;
        const auto& host_side_buffer = input_pair.second;
        const auto& device_side_buffer = context.buffers.device_side.inputs.at(name);
        copy_buffer_to_device(context, name, device_side_buffer, host_side_buffer);

    }

    spdlog::debug("Copying in-out buffers to the GPU (to pristine, not-to-be-altered copies).");
    for(const auto& buffer_name : buffer_names(*context.kernel_adapter_,parameter_direction_t::inout)  ) {
        auto& host_side_buffer = context.buffers.host_side.inputs.at(buffer_name);
        const auto& device_side_buffer = context.buffers.device_side.inputs.at(buffer_name);
        copy_buffer_to_device(context, buffer_name, device_side_buffer, host_side_buffer);
    }
    if (context.options.sync_after_buffer_op)
    gpu_sync(context);
}

void copy_buffer_to_host(
    execution_ecosystem_t      ecosystem,
    cl::CommandQueue*          opencl_queue,
    const device_buffer_type&  device_side_buffer,
    host_buffer_t&          host_side_buffer)
{
    if (ecosystem == execution_ecosystem_t::cuda) {
        cuda::memory::copy(host_side_buffer.data(), device_side_buffer.cuda.data(), host_side_buffer.size());
    } else {
        // OpenCL
        const constexpr auto blocking { CL_TRUE };
        constexpr const auto no_offset { 0 };
        opencl_queue->enqueueReadBuffer(device_side_buffer.opencl, blocking, no_offset, host_side_buffer.size(), host_side_buffer.data());
    }
}

// Note: must take the context as non-const, since it has vector members, and vectors
// are value-types, not reference-types, i.e. copying into those vectors changes
// the context.
void copy_outputs_from_device(execution_context_t& context)
{
    spdlog::debug("Copying outputs back to host memory.");
    for(auto& output_pair : context.buffers.host_side.outputs) {
        const auto& name = output_pair.first;
        auto& host_side_buffer = output_pair.second;
        const auto& device_side_buffer = context.buffers.device_side.outputs.at(name);
        spdlog::trace("Copying device output buffer to host output buffer for '{}'", name);
        copy_buffer_to_host(
            context.ecosystem,
            &context.opencl.queue,
            device_side_buffer,
            host_side_buffer);
    }
    gpu_sync(context);
}

device_buffer_type create_device_side_buffer(
    const std::string&                      name,
    std::size_t                             size,
    execution_ecosystem_t                   ecosystem,
    const optional<const cuda::context_t>&  cuda_context,
    optional<const cl::Context>             opencl_context)
{
    device_buffer_type result;
    if (ecosystem == execution_ecosystem_t::cuda) {
        auto region = cuda::memory::device::allocate(*cuda_context, size);
        memory_region sp {static_cast<byte_type*>(region.data()), region.size() };
        spdlog::trace("Created GPU-side buffer for '{}': {} bytes at {}", name, sp.size(), (void*) sp.data());
        result.cuda = sp;
    }
    else { // OpenCL
        cl::Buffer buffer { opencl_context.value(), CL_MEM_READ_WRITE, size };
            // TODO: Consider separating in, out and in/out buffer w.r.t. OpenCL creating, to be able to pass
            // other flags.
        spdlog::trace("Created an OpenCL read/write buffer with size {} for kernel parameter '{}'", size, name);
        result.opencl = std::move(buffer);
    }
    return result;
}

device_buffers_map create_device_side_buffers(
    execution_ecosystem_t                   ecosystem,
    const optional<const cuda::context_t>&  cuda_context,
    optional<cl::Context>                   opencl_context,
    const buffer_size_map                   buffer_sizes)
{
    return util::transform<device_buffers_map>(
        buffer_sizes,
        [&](const auto& p) {
            const auto& name = p.first;
            const auto& size = p.second;
            spdlog::debug("Creating GPU-side buffer for '{}' of size {} bytes.", name, size);
            auto buffer = create_device_side_buffer(
                name, size,
                ecosystem,
                cuda_context,
                opencl_context);
            return device_buffers_map::value_type { name, std::move(buffer) };
        } );
}

device_buffers_map create_device_side_buffers(
    execution_ecosystem_t                   ecosystem,
    const optional<const cuda::context_t>&  cuda_context,
    optional<cl::Context>                   opencl_context,
    const host_buffers_t&                   host_side_buffers)
{
    auto buffer_sizes = util::transform<buffer_size_map>(
        host_side_buffers,
        [&](const auto& p) -> std::pair<string, size_t>
        { return { p.first, p.second.size() }; });
    return create_device_side_buffers(ecosystem, cuda_context, opencl_context, buffer_sizes);
}

device_buffers_map create_scratch_buffers(execution_context_t& context)
{
    auto scratch_buffer_details = util::filter(
        context.get_kernel_adapter().buffer_details(),
        [&](const auto& buffer_details) {
            return buffer_details.direction == parameter_direction_t::scratch;
        });
    auto buffer_sizes = util::transform<buffer_size_map>(
        scratch_buffer_details,
        [&](const auto& buffer_details) -> std::pair<string, size_t>
        { return { buffer_details.name, apply_size_calc(buffer_details.size_calculator, context) }; });
    return create_device_side_buffers(context.ecosystem, context.cuda.context, context.opencl.context, buffer_sizes);
}

// Notes:
// 1. Why is cuda_stream not an optional-ref? Because optional ref's are
//    problematic in C++
// 2. This function never performs a sync, as it is too low-level to know
//    whether or not it needs to
void schedule_zero_buffer(
    execution_ecosystem_t                  ecosystem,
    const device_buffer_type               buffer,
    const optional<const cuda::stream_t*>  cuda_stream,
    const cl::CommandQueue*                opencl_queue,
    const std::string&                     buffer_name)
{
    spdlog::trace("Scheduling the zeroing of GPU-side output buffer for '{}'", buffer_name);
    if (ecosystem == execution_ecosystem_t::cuda) {
        cuda_stream.value()->enqueue.memzero(buffer.cuda.data(), buffer.cuda.size());
    } else {
        // OpenCL
        const constexpr unsigned char zero_pattern { 0 };
        const constexpr size_t no_offset { 0 };
        size_t size;
        buffer.opencl.getInfo(CL_MEM_SIZE, &size);
        opencl_queue->enqueueFillBuffer(buffer.opencl, zero_pattern, no_offset, size);
    }
}

void schedule_zero_output_buffers(execution_context_t& context)
{
    const auto& ka = *context.kernel_adapter_;
    auto output_only_buffers = buffer_names(ka, parameter_direction_t::out);
    if (output_only_buffers.empty()) {
        spdlog::debug("There are no output-only buffers to fill with zeros.");
        return;
    }
    spdlog::debug("Scheduling the zeroing of GPU-side output-only buffers.");
    for(const auto& buffer_name : output_only_buffers) {
        const auto& buffer = context.buffers.device_side.outputs.at(buffer_name);
        // Note: A bit of a kludge, but - we can't copy the optional, since stream copying is verbotten,
        // and we can't use an optional<stream_t&>, since C++ doesn't like optional-of-references
        auto maybe_cuda_stream_ptr = context.cuda.stream ? optional<cuda::stream_t*>(&context.cuda.stream.value()) : nullopt;
        schedule_zero_buffer(context.ecosystem, buffer, maybe_cuda_stream_ptr, &context.opencl.queue, buffer_name);
    }
    if (context.options.sync_after_buffer_op) {
        gpu_sync(context);
        spdlog::debug("GPU-side Output-only buffers filled with zeros.");
    }
}

void schedule_zero_single_buffer(const execution_context_t& context, const device_buffer_type& buffer)
{
    // Note: A bit of a kludge, but - we can't copy the optional, since stream copying is verbotten,
    // and we can't use an optional<stream_t&>, since C++ doesn't like optional-of-references
    auto maybe_cuda_stream_ptr = context.cuda.stream ? optional<const cuda::stream_t*>(&context.cuda.stream.value()) : nullopt;
    schedule_zero_buffer(context.ecosystem, buffer, maybe_cuda_stream_ptr, &context.opencl.queue,
                         "kernel_runner_L2_cache_clearing_gadget");
    if (context.options.sync_after_buffer_op) { gpu_sync(context); }
}

static size_t get_l2_cache_size(const execution_context_t& context)
{
    if (context.ecosystem == execution_ecosystem_t::cuda) {
        return context.cuda.context->device().get_attribute(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE);
    }
    else { // opencl
        constexpr const size_t opencl_device_max_cache_size_mibytes { 48 };
        spdlog::warn("OpenCL does not support determining L2 cache size; assuming it is no larger than {} MiB", opencl_device_max_cache_size_mibytes);
        return opencl_device_max_cache_size_mibytes * 1024l * 1024l;
    }
}

void create_device_side_buffers(execution_context_t& context)
{
    spdlog::debug("Creating {} GPU-side buffers.", context.buffers.host_side.inputs.size());
    context.buffers.device_side.inputs = create_device_side_buffers(
        context.ecosystem,
        context.cuda.context,
        context.opencl.context,
        context.buffers.host_side.inputs);
    spdlog::debug("Input buffers, and pristine copies of in-out buffers, created in GPU memory.");
    context.buffers.device_side.outputs = create_device_side_buffers(
        context.ecosystem,
        context.cuda.context,
        context.opencl.context,
        context.buffers.host_side.outputs);
            // ... and remember the behavior regarding in-out buffers: For each in-out buffers, a buffer
            // is created in _both_ previous function calls
    spdlog::debug("Output buffers, and a work copy of the input buffers, have created in GPU memory");
    context.buffers.device_side.scratch = create_scratch_buffers(context);
    spdlog::debug("Scratch buffers created in GPU memory");
    if (context.options.clear_l2_cache) {
        auto size = get_l2_cache_size(context);
        context.buffers.device_side.l2_cache_clearing_gadget.emplace(create_device_side_buffer(
            "kernel_runner_L2_cache_clearing_gadget", size,
            context.ecosystem,
            context.cuda.context,
            context.opencl.context));
    }
    gpu_sync(context);
}

// Note: Will create buffers also for each inout buffers
void create_host_side_output_buffers(execution_context_t& context)
{
    // TODO: Double-check that all output and inout buffers have entries in the map we've received.

    auto& all_params = context.kernel_adapter_->parameter_details();
    auto output_buffer_details = util::filter(all_params,
        [&](const auto& param_details) {
            return is_output(param_details.direction) and param_details.kind == kernel_parameters::kind_t::buffer;
        }
    );
    spdlog::debug("Creating {} host-side output buffers", output_buffer_details.size());

    context.buffers.host_side.outputs = util::transform<decltype(context.buffers.host_side.outputs)>(
        output_buffer_details,
        [&](const kernel_adapter::single_parameter_details& buffer_details) {
            auto buffer_name = buffer_details.name;
            size_t buffer_size;
            if (buffer_details.direction == parameter_direction_t::inout) {
                buffer_size = context.buffers.host_side.inputs.at(buffer_name).size();
                // TODO: What if an output size has been specified on the command line
            }
            else if (util::contains(context.options.output_buffer_sizes, std::string{buffer_name})) {
                buffer_size = context.options.output_buffer_sizes.at(buffer_name);
            }
            else {
                if (buffer_details.size_calculator == nullptr) {
                    spdlog::critical("Cannot determine the size of output buffer '{}': No used-specified size and no size calculator available", buffer_name);
                    throw std::logic_error(
                        "Cannot determine the size of output buffer '" + std::string{buffer_name}
                        + "': No user-specified size and no size calculator function available");
                }
                buffer_size = buffer_details.size_calculator(
                    context.buffers.host_side.inputs,
                    context.scalar_input_arguments.typed,
                    context.preprocessor_definitions.finalized.valueless,
                    context.preprocessor_definitions.finalized.valued,
                    context.options.forced_launch_config_components);
            }
            auto host_side_output_buffer = host_buffer_t(buffer_size);
            spdlog::trace("Created a host-side output buffer of size {} for kernel parameter '{}'", buffer_size,  buffer_name);
            return std::make_pair(buffer_details.name, std::move(host_side_output_buffer));
        }
    );
}

void schedule_reset_of_inout_buffers_working_copy(execution_context_t& context)
{
    auto& ka = *context.kernel_adapter_;
    auto inout_buffer_names = buffer_names(ka, parameter_direction_t::inout);
    if (inout_buffer_names.empty()) {
        return;
    }
    spdlog::debug("Scheduling an initialization of the work-copies of the in-out buffers with the pristine copies.");
    for(const auto& inout_buffer_name : inout_buffer_names) {
        const auto& pristine_copy = context.buffers.device_side.inputs.at(inout_buffer_name);
        const auto& work_copy = context.buffers.device_side.outputs.at(inout_buffer_name);
        spdlog::debug("Initializing work-copy of inout buffer '{}'", inout_buffer_name);
        copy_buffer_on_device(context.ecosystem,
            context.ecosystem == execution_ecosystem_t::opencl ? &context.opencl.queue : nullptr,
            work_copy, pristine_copy);
    }
    if (context.options.sync_after_buffer_op) {
        gpu_sync(context);
    }
}

void write_data_to_file(
    std::string          kind,
    std::string          name,
    const_memory_region  data,
    filesystem::path     destination,
    bool                 overwrite_allowed,
    bool                 log_at_info_level)
{
    auto level = log_at_info_level ? spdlog::level::info : spdlog::level::debug;
    spdlog::log(level, "Writing {} '{}' to file {}", kind, name, destination.c_str());
    util::write_data_to_file(kind + " '" + name + "'", data, destination, overwrite_allowed);
}

void write_buffers_to_files(execution_context_t& context)
{
    spdlog::info("Writing output buffers to files.");
    // Unfortunately, decent ranged-for iteration on maps is only possible with C++17
    for(const auto& pair : context.buffers.host_side.outputs) {
        const auto& buffer_name = pair.first;
        const auto& buffer = pair.second;
        auto write_destination = maybe_prepend_base_dir(
            context.options.buffer_base_paths.output,
            context.buffers.filenames.outputs[buffer_name]);
        bool dont_log_at_info_level { false };
        write_data_to_file(
            "output buffer", buffer_name, as_region(buffer), write_destination,
            context.options.overwrite_allowed, dont_log_at_info_level);
    }
}
