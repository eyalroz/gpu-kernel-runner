#include "buffer_ops.hpp"

#include "kernel_adapter.hpp"
#include "util/buffer_io.hpp"
#include "opencl-related/miscellany.hpp"

#include "util/spdlog-extra.hpp"
#include <spdlog/spdlog.h>

using std::size_t;
using std::string;

// Notes:
// * This function does not utilize, nor set, the expected/intended buffer sizes;
//   it merely reads what's actually in the buffer files
// * Not using memory-mapping, this is a plain read
// * No need for any special treatment for image-type kernel arguments
static host_buffers_t read_input_buffers_from_files(
    const name_set&           buffer_names,
    const maybe_string_map&   filenames,
    const filesystem::path&   buffer_directory)
{
    spdlog::debug("Reading input buffers from files.");

    host_buffers_t result;
    for(const auto& name : buffer_names) {
        // Note: Even though the map is of optional<string>, we expect all relevant buffers
        // to have already had names determined
        auto path = maybe_prepend_base_dir(buffer_directory, filenames.at(name).value());
        try {
            spdlog::debug("Reading buffer '{}' from {}", name, path.native());
            host_buffer_t buffer = util::read_input_file(path);
            spdlog::debug("Have read buffer '{}': {} bytes from {}", name, buffer.size(), path.native());
            spdlog::trace("... into a buffer at {}", static_cast<void*>(buffer.data()));
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
    // TODO: Check that we don't need to re-check that these are actually buffers
    // Note: The input buffers don't need any distinguishing treatment to
    // images vs raw buffers
    auto input_buffer_names = buffer_names(*context.kernel_adapter_,
        [&](const auto& spd) { return is_input(spd.direction); } );
    context.buffers.host_side.inputs = read_input_buffers_from_files(
        input_buffer_names,
        context.buffers.filenames.inputs,
        context.options.buffer_base_paths.input);
}

// Notes:
// * This function is synchronous and not affected by the sync_after_buffer_op option
// * This works for raw buffers and for images
void copy_buffer_to_device(
    const execution_context_t& context,
    const std::string&         buffer_name,
    const device_buffer_type&  device_side_buffer,
    const host_buffer_t&       host_side_buffer)
{
    spdlog::debug("Copying buffer '{}' (size {} bytes) from host to device memory",
        buffer_name, host_side_buffer.size());
    if (context.ecosystem == execution_ecosystem_t::cuda) {
        spdlog::debug("CUDA copy of {} : host-side {} -> device-side {}",
            host_side_buffer.size(), (void *) host_side_buffer.data(),
            (void *) device_side_buffer.cuda.data());
        cuda::context::current::scoped_override_t scoped_context_override{*context.cuda.context};
        cuda::memory::copy(device_side_buffer.cuda.data(), host_side_buffer.data(), host_side_buffer.size());
    } else {
        // OpenCL
        const constexpr auto is_blocking{ CL_TRUE };
        if (device_side_buffer.opencl.is_image) {
            auto &image = device_side_buffer.opencl.image();
            auto ndims = device_side_buffer.opencl.ndims();
            static const cl::size_t<3> no_offset{}; // this is guaranteed to be initialized to 0's
            cl::size_t<3> region;
            region[0] = device_side_buffer.opencl.dims[0];
            region[1] = (ndims >= 2) ? device_side_buffer.opencl.dims[1] : 1;
            region[2] = (ndims >= 3) ? device_side_buffer.opencl.dims[2] : 1;
            // Q: Why are we using 1 rather than 0?
            // A: It's required apparently, see:
            // https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clEnqueueWriteImage.html
            spdlog::debug("Enqueuing copy of OpenCL image of dimensions {}, with pitches {}, {} and host side data {}",
                format_as(device_side_buffer.opencl.dims),
                device_side_buffer.opencl.pitches[0],
                device_side_buffer.opencl.pitches[1],
                (void *)(host_side_buffer.data()));
            // Note: Assuming we are ok w.r.t. memory alignment
            context.opencl.queue.enqueueWriteImage(image,
                is_blocking, no_offset, region,
                device_side_buffer.opencl.pitches[0],
                device_side_buffer.opencl.pitches[1],
                const_cast<char *>(host_side_buffer.data())
            );
        } else {
            constexpr const size_t no_offset{0};
            spdlog::debug("Enqueuing copy of host-side to device-side buffer for '{}'", buffer_name);
            context.opencl.queue.enqueueWriteBuffer(
                device_side_buffer.opencl.raw, is_blocking, no_offset,
                host_side_buffer.size(), host_side_buffer.data());
        }
    }
}

void copy_buffer_on_device(
    execution_ecosystem_t      ecosystem,
    cl::CommandQueue*          queue,
    const device_buffer_type&  destination,
    const device_buffer_type&  source)
{
    if (ecosystem == execution_ecosystem_t::cuda) {
        cuda::memory::copy(destination.cuda.data(), source.cuda.data(), destination.cuda.size());
    } else { // OpenCL
        if (source.opencl.is_image != destination.opencl.is_image) {
            die("Incompatible source and destination buffers for a copy");
        }
        if (source.opencl.is_image) {
            cl::size_t<3> no_offset {}; // will be initialized to 0
            queue->enqueueCopyImage(
                source.opencl.image(), destination.opencl.image(),
                no_offset, no_offset, source.opencl.dims);
        }
        else {
            constexpr const auto no_offset { 0 };
            size_t size;
            source.opencl.raw.getInfo(CL_MEM_SIZE, &size);
            queue->enqueueCopyBuffer(source.opencl.raw, destination.opencl.raw, no_offset, no_offset, size);
        }
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
    cl::CommandQueue* const    opencl_queue,
    const device_buffer_type&  device_side_buffer,
    host_buffer_t&             host_side_buffer)
{
    if (ecosystem == execution_ecosystem_t::cuda) {
        cuda::memory::copy(host_side_buffer.data(), device_side_buffer.cuda.data(), host_side_buffer.size());
    } else {
        // OpenCL
        const constexpr auto is_blocking { CL_TRUE };
        if (device_side_buffer.opencl.is_image) {
            auto& image = device_side_buffer.opencl.image();
            const cl::size_t<3> no_offset{}; // this is guaranteed to be initialized to 0's
            opencl_queue->enqueueReadImage(image,
                is_blocking, no_offset,
                device_side_buffer.opencl.dims,
                device_side_buffer.opencl.pitches[0],
                device_side_buffer.opencl.pitches[1],
                const_cast<char *>(host_side_buffer.data())
            );
        }
        else {
            constexpr const auto no_offset { 0 };
            opencl_queue->enqueueReadBuffer(device_side_buffer.opencl.raw,
                is_blocking, no_offset, host_side_buffer.size(),
                host_side_buffer.data());
        }
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

static device_buffer_type create_device_side_buffer(
    const std::string&                      name,
    device_side_buffer_info_t               buffer_info,
    execution_ecosystem_t                   ecosystem,
    const optional<const cuda::context_t>&  cuda_context,
    const optional<const cl::Context>&      opencl_context)
{
    device_buffer_type result;
    if (ecosystem == execution_ecosystem_t::cuda) {
        if (buffer_info.is_image) { throw std::runtime_error("Images in CUDA kernels are not currently supported."); }
        auto region = cuda::memory::device::allocate(*cuda_context, buffer_info.size);
        memory_region sp {static_cast<byte_type*>(region.data()), region.size() };
        spdlog::trace("Created GPU-side buffer for '{}': {} bytes at {}", name, sp.size(), (void*) sp.data());
        result.cuda = sp;
    }
    else { // OpenCL
        result.opencl.is_image = buffer_info.is_image;
        if (buffer_info.is_image) {
            result.opencl.is_image = true;
            auto ndims = buffer_info.dimensions.size();
            result.opencl.dims[0] = buffer_info.dimensions[0];
            result.opencl.dims[1] = (ndims >= 2) ? buffer_info.dimensions[1] : 0;
            result.opencl.dims[2] = (ndims >= 3) ? buffer_info.dimensions[2] : 0;
            // Note that we may need to use 1 rather than 0 in some cases, see:
            // https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clEnqueueWriteImage.html
            cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_HOST_WRITE_ONLY;
                // perhaps only CL_MEM_READ ?
                // perhaps make it depend on the kernel signature?
            cl_channel_order channel_order = get_image_channel_order(buffer_info);
            cl_channel_type channel_type = get_image_channel_type(buffer_info);
            // Note: One must _always_ specify pitch values, with 0 meaning no value,
            // or rather, the exact amount necessary to fit the data into memory with
            // no slack/padding.
            auto pitches_ = buffer_info.pitches.value_or(dimensions_t{0, 0, 0});
            auto image_format = cl::ImageFormat(channel_order, channel_type);
            switch(ndims) {
            case 2:
                result.opencl.image2d = cl::Image2D(*opencl_context, flags,
                    image_format,
                    buffer_info.dimensions[0],
                    buffer_info.dimensions[1],
                    pitches_[0]);
                break;
            case 3:
                result.opencl.image3d = cl::Image3D(*opencl_context, flags,
                    image_format,
                    buffer_info.dimensions[0],
                    buffer_info.dimensions[1],
                    buffer_info.dimensions[2],
                    pitches_[0],
                    pitches_[1]);
                break;
            default: throw std::invalid_argument("Only 2D and 3D OpenCL images are supported, but a " +
                    std::to_string(ndims) + "D image has been requested");
            }
        }
        else { // it's a raw buffer
            cl::Buffer buffer { opencl_context.value(), CL_MEM_READ_WRITE, buffer_info.size };
                // TODO: Consider separating in, out and in/out buffer w.r.t. OpenCL creating, to be able to pass
                // other flags.
            spdlog::trace("Created an OpenCL read/write buffer with size {} for kernel argument '{}'", buffer_info.size, name);
            result.opencl.raw = std::move(buffer);
        }
    }
    return result;
}

static device_buffers_map create_device_side_buffers(
    execution_ecosystem_t                   ecosystem,
    const optional<const cuda::context_t>&  cuda_context,
    optional<cl::Context>                   opencl_context,
    const device_buffer_info_map_t&         device_buffers_info)
{
    return util::transform<device_buffers_map>(
        device_buffers_info,
        [&](const auto& p) {
            const auto& name = p.first;
            const device_side_buffer_info_t& sdbi = p.second;
            spdlog::debug("Creating GPU-side {} for '{}' ", sdbi.is_image ? "image" : "buffer", name);
            auto buffer = create_device_side_buffer(
                name, sdbi,
                ecosystem,
                cuda_context,
                opencl_context);
            return device_buffers_map::value_type { name, std::move(buffer) };
        } );
}

// Note: Can only be called for non-scratch buffers, and only after
// the host-side buffers have
device_side_buffer_info_t get_device_side_buffer_info(
    kernel_adapter::single_parameter_details const& spd, execution_context_t const& context)
{
    if (spd.kind != kernel_parameters::kind_t::buffer) {
        throw std::invalid_argument("Got a non-buffer parameter info struct when a buffer was expected");
    }
    if (spd.direction == parameter_direction_t::scratch) {
        throw std::invalid_argument("This function does not (currently) support determining scratch buffer sizes");
    }
    device_side_buffer_info_t result;
    result.is_image = (spd.buffer_kind == buffer_kind_t::image);
    if (not result.is_image) {
        if (is_input(spd)) { result.size = context.buffers.host_side.inputs.at(spd.name).size(); }
        else {
            auto maybe_size = resolve_buffer_size(spd, context);
            if (not maybe_size) {
                // Can this even happen at this point? It shouldn't be possible... we should
                // have verified everything
                die("Cannot determine the size of output buffer '{}'", spd.name);;
            }
            result.size = *maybe_size;
        }
    }
    else {
        result.pitches = util::safe_lookup(context.buffer_pitches, spd.name);
        result.dimensions = context.buffer_dimensions.at(spd.name);
        result.num_channels = spd.num_channels;
        result.channel_elements_type = spd.element_type;
        // At this point we assume we have sufficient information about the image to calculate its size
        result.size = *resolve_buffer_size(spd, context);
    }
    return result;
}

device_buffer_info_map_t get_device_buffers_info_map(
    name_set const& buffer_names,
    execution_context_t const& context)
{
    auto all_buffer_details = context.kernel_adapter_->all_buffer_details();
    auto extract_name = [&](auto const& spd) { return std::string{spd.name}; };
    auto all_buffer_details_map = util::to_map<std::unordered_map>(all_buffer_details, extract_name);
    // Should be equivalent to:
    //    util::transform<std::unordered_map<string, kernel_adapter::single_parameter_details>>(
    //    all_buffer_details, [&](auto const& spd) { return std::make_pair(spd.name, spd); });

    return util::transform<device_buffer_info_map_t>(buffer_names,
        [&](auto const& buffer_name) {
            auto maybe_spd = util::safe_lookup(all_buffer_details_map, buffer_name);
            if (not maybe_spd) {
                die("Could not find parameter info for kernel buffer parameter {}", buffer_name);
            }
            auto device_side_buffer_info = get_device_side_buffer_info(*maybe_spd, context);
            return make_pair(buffer_name, device_side_buffer_info);
        });
//    device_buffer_info_map_t device_buffer_info_map {};
}

device_buffers_map create_device_side_buffers(
    execution_context_t const& context,
    const host_buffers_t& host_side_buffers)
{
    auto host_side_buffer_names = util::keys(host_side_buffers);
    spdlog::debug("Creating device-side buffers: {}", host_side_buffer_names);
    auto device_side_buffer_info_map = get_device_buffers_info_map(host_side_buffer_names, context);
    return create_device_side_buffers(
        context.ecosystem,
        context.cuda.context,
        context.opencl.context,
        device_side_buffer_info_map);
}

static device_buffers_map create_all_device_scratch_buffers(execution_context_t& context)
{
    auto scratch_buffer_names = buffer_names(*context.kernel_adapter_, is_scratch);
    auto buffer_info_map = get_device_buffers_info_map(scratch_buffer_names, context);
    return create_device_side_buffers(context.ecosystem, context.cuda.context, context.opencl.context, buffer_info_map);
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
    const std::string&                     buffer_name,
    const device_side_buffer_info_t&       buffer_info)
{
    spdlog::trace("Scheduling the zeroing of GPU-side output buffer for '{}'", buffer_name);
    if (ecosystem == execution_ecosystem_t::cuda) {
        cuda_stream.value()->enqueue.memzero(buffer.cuda.data(), buffer.cuda.size());
    } else {
        // OpenCL
        if (buffer.opencl.is_image) {
//                auto dims = context.options.buffer_dimensions.at(buffer_name);
//                auto pitches = context.options.buffer_pitches.at(buffer_name);
                bool image_is_2d = (buffer_info.dimensions.size() == 2);
                auto& image = image_is_2d ?
                              (cl::Image&) buffer.opencl.image2d :
                              (cl::Image&) buffer.opencl.image3d;
                cl::size_t<3> dims_;
                dims_[0] = buffer_info.dimensions[0];
                dims_[1] = buffer_info.dimensions[1];
                dims_[2] = image_is_2d ? 0 : buffer_info.dimensions[2];
                cl_int4 color = { 0, 0, 0, 0 };
                cl::size_t<3> origin; // it's initialized to zeros; and we can't init it ourselves
                opencl_queue->enqueueFillImage(image, color, origin, dims_);
        }
        else {
            const constexpr unsigned char zero_pattern { 0 };
            const constexpr size_t no_offset { 0 };
            size_t size;
            buffer.opencl.raw.getInfo(CL_MEM_SIZE, &size);
            opencl_queue->enqueueFillBuffer(buffer.opencl.raw, zero_pattern, no_offset, size);
        }
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
    auto buffer_info_map = get_device_buffers_info_map(output_only_buffers, context);

    spdlog::debug("Scheduling the zeroing of GPU-side output-only buffers.");
    for(const auto& buffer_name : output_only_buffers) {
        const auto& buffer = context.buffers.device_side.outputs.at(buffer_name);
        // Note: A bit of a kludge, but - we can't copy the optional, since stream copying is verbotten,
        // and we can't use an optional<stream_t&>, since C++ doesn't like optional-of-references
        // TODO: Switch to optional_ref; we can get it, for example, from the CUDA API wrappers
        auto maybe_cuda_stream_ptr = context.cuda.stream ? optional<cuda::stream_t*>(&context.cuda.stream.value()) : nullopt;
        schedule_zero_buffer(context.ecosystem, buffer, maybe_cuda_stream_ptr, &context.opencl.queue, buffer_name, buffer_info_map.at(buffer_name));
    }
    if (context.options.sync_after_buffer_op) {
        gpu_sync(context);
        spdlog::debug("GPU-side Output-only buffers filled with zeros.");
    }
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

void schedule_zero_single_raw_buffer(const execution_context_t& context, const device_buffer_type& buffer)
{
    // Note: A bit of a kludge, but - we can't copy the optional, since stream copying is verbotten,
    // and we can't use an optional<stream_t&>, since C++ doesn't like optional-of-references
    auto maybe_cuda_stream_ptr = context.cuda.stream ? optional<const cuda::stream_t*>(&context.cuda.stream.value()) : nullopt;
    auto dsbi = make_raw_device_side_buffer_info(get_l2_cache_size(context));
    schedule_zero_buffer(
        context.ecosystem, buffer, maybe_cuda_stream_ptr, &context.opencl.queue,
        "kernel_runner_L2_cache_clearing_gadget", dsbi);
    if (context.options.sync_after_buffer_op) { gpu_sync(context); }
}

void create_all_device_side_buffers(execution_context_t& context)
{
    spdlog::debug("Creating {} GPU-side input buffers (including pristine copies of in-out buffers).",
        context.buffers.host_side.inputs.size());
    context.buffers.device_side.inputs = create_device_side_buffers(context, context.buffers.host_side.inputs);
    spdlog::debug("Input buffers, and pristine copies of in-out buffers, now all created in GPU memory.");
    spdlog::debug("Creating {} GPU-side output buffers (including work copies of in-out buffers).",
        context.buffers.host_side.outputs.size());
    context.buffers.device_side.outputs = create_device_side_buffers(context, context.buffers.host_side.outputs);
        // ... and remember the behavior regarding in-out buffers: For each in-out buffers, a buffer
        // is created in _both_ previous function calls
    spdlog::debug("Output buffers, including work copies of in-out buffers, now all created in GPU memory");
    context.buffers.device_side.scratch = create_all_device_scratch_buffers(context);
    spdlog::debug("Scratch buffers now all created in GPU memory");
    if (context.options.clear_l2_cache) {
        auto sdbi = make_raw_device_side_buffer_info(get_l2_cache_size(context));
        context.buffers.device_side.l2_cache_clearing_gadget.emplace(create_device_side_buffer(
            "kernel_runner_L2_cache_clearing_gadget", sdbi,
            context.ecosystem,
            context.cuda.context,
            context.opencl.context));
    }
    gpu_sync(context);
}

size_t get_output_buffer_size(const execution_context_t &context,
    const kernel_adapter::single_parameter_details &buffer_details, const char *&buffer_name)
{
    // maybe we've already computed this?
    auto already_existing = util::safe_lookup(context.buffers.host_side.outputs, buffer_name);
    if (already_existing) {
        return already_existing->size();
    }
    // The following should only work for inout buffers; should we double-check?
    auto as_input_buffer = util::safe_lookup(context.buffers.host_side.inputs, buffer_name);
    auto explicitly_specified_size = util::safe_lookup(context.output_buffer_sizes, std::string{buffer_name});
    if (as_input_buffer and explicitly_specified_size) {
        // TODO: Can this sanity check be moved to when the input buffers are read from files?
        if (as_input_buffer->size() != *explicitly_specified_size) {
            die("Explicitly-specified size of output buffer {} and its input file size disagree: {} != {}",
                buffer_name, as_input_buffer->size(), *explicitly_specified_size);
        }
    }
    if (as_input_buffer) { return as_input_buffer->size(); }
    if (explicitly_specified_size) { return *explicitly_specified_size; }
    auto calculated = resolve_buffer_size(buffer_details, context);
    if (calculated) { return *calculated; }
    die("Cannot determine the size of output buffer '{}': No user-specified size "
        "and no size calculator is available", buffer_name);
}

// Notes:
// * Will create buffers also for inout parameters, not just output-only
// * No need for special treatment of image-type kernel arguments
void create_host_side_output_buffers(execution_context_t& context)
{
    // TODO: Double-check that all output and inout buffers have entries in the map we've received.

    auto& all_params = context.kernel_adapter_->parameter_details();
    auto output_buffer_details = util::filter(all_params,
        [&](const auto& spd) { return is_output(spd) and is_buffer(spd); });
    spdlog::debug("Creating {} host-side output buffers", output_buffer_details.size());

    context.buffers.host_side.outputs = util::transform<host_buffers_t>(
        output_buffer_details,
        [&](const kernel_adapter::single_parameter_details& buffer_details) {
            auto buffer_name = buffer_details.name;
            auto buffer_size = get_output_buffer_size(context, buffer_details, buffer_name);
            auto host_side_output_buffer = host_buffer_t(buffer_size);
            spdlog::trace("Created a host-side output buffer of size {} for kernel argument '{}'", buffer_size,  buffer_name);
            return std::make_pair(buffer_details.name, std::move(host_side_output_buffer));
        }
    );
}

void schedule_reset_of_inout_buffers_working_copy(execution_context_t& context)
{
    auto& ka = *context.kernel_adapter_;
    auto inout_buffer_names = buffer_names(ka, parameter_direction_t::inout);
    if (inout_buffer_names.empty()) {
        spdlog::trace("There are no output-only buffers to reset to zero.");
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
    std::string              kind,
    std::string              name,
    const_memory_region      data,
    const filesystem::path&  destination,
    bool                     overwrite_allowed,
    bool                     log_at_info_level)
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
