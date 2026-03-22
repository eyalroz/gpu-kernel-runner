
#include "kernel_adapter.hpp"
#include "util/spdlog-extra.hpp"

std::ostream& operator<<(std::ostream& os, cuda::grid::dimensions_t dims)
{
    return os << '(' << dims.x << " x " << dims.y << " x " << dims.z << " x " << ')';
}

kernel_adapter::single_parameter_details
kernel_adapter::single_parameter_details::aliases(
    std::initializer_list<const char*> extra_aliases) const
{
    auto result = *this;
    std::copy(std::cbegin(extra_aliases), std::cend(extra_aliases), std::back_inserter(result.aliases_));
    return result;
}

kernel_adapter::single_parameter_details
kernel_adapter::single_parameter_details::alias(const char* extra_alias) const
{
    auto result = *this;
    result.aliases_.emplace_back(extra_alias);
    return result;
}

kernel_adapter::single_parameter_details
kernel_adapter::single_parameter_details::alias(const std::string& extra_alias) const
{
    return alias(extra_alias.c_str());
}

bool kernel_adapter::single_parameter_details::has_alias(const std::string& alias) const
{
    return util::find(aliases_,alias) != aliases_.cend();
}

kernel_adapter::parameter_details_type kernel_adapter::scalar_parameter_details() const
{
    parameter_details_type all_params = parameter_details();
    return util::filter(all_params, [](const single_parameter_details& param) { return param.kind == scalar; });
}

kernel_adapter::parameter_details_type kernel_adapter::all_buffer_details() const
{
    parameter_details_type all_params = parameter_details();
    return util::filter(all_params, [](const single_parameter_details& param) { return param.kind == buffer; });
}

// kernel_adapter::parameter_details_type kernel_adapter::all_image_details() const
// {
//     parameter_details_type all_params = parameter_details();
//     return util::filter(all_params, [](const single_parameter_details& param) {
//         return param.kind == kernel_parameters::kind_t::buffer and
//              param.buffer_kind == buffer_kind_t::image;
//     });
// }

optional_launch_config_components_t kernel_adapter::deduce_launch_config(const execution_context_t& context) const
{
    auto components = context.options.forced_launch_config_components;
    if (not components.dynamic_shared_memory_size) {
        components.dynamic_shared_memory_size = 0;
    }
    if (components.is_sufficient()) {
        return components;
    }

    throw std::runtime_error(
        "Unable to deduce launch configuration - please specify all launch configuration components "
        "explicitly using the command-line");
}

optional_launch_config_components_t kernel_adapter::make_launch_config(const execution_context_t& context) const {
    auto& forced = context.options.forced_launch_config_components;
    if (forced.is_sufficient()) {
        return forced;
    }
    else return deduce_launch_config(context);
}

kernel_adapter::single_parameter_details kernel_adapter::raw_buffer_details(
    const char*            name,
    parameter_direction_t  direction,
    size_calculator_type   size_calculator,
    bool                   required_,
    const std::initializer_list<std::string>& name_aliases)
{
    if ((direction == scratch) and (size_calculator == no_size_calc)) {
        throw std::invalid_argument("Scratch buffer parameters must be defined with a size calculator");
    }
    return single_parameter_details {
        name, name_aliases, buffer, no_parser, size_calculator,
        no_pusher, direction, required_, buffer_kind_t::raw};
}


kernel_adapter::single_parameter_details kernel_adapter::image_details(
    const char*            name,
    size_t                 num_dimensions,
    parameter_direction_t  direction,
    size_t                 num_channels,
    kernel_parameters::element_type_descriptor_t
                           element_type,
    bool                   required,
    const std::initializer_list<std::string>& name_aliases)
{
    if (direction == scratch) {
        throw std::invalid_argument("Scratch image buffers are not supported");
    }
    return single_parameter_details {
        name, name_aliases, buffer, no_parser, no_size_calc,
        no_pusher, direction, required, buffer_kind_t::image,
        element_type, num_channels, num_dimensions };
}

kernel_adapter::single_parameter_details kernel_adapter::image_details(
    const char*            name,
    size_t                 num_dimensions,
    parameter_direction_t  direction,
    size_t                 num_channels,
    const std::string&     element_type_name,
    bool                   required,
    const std::initializer_list<std::string>& name_aliases)
{
    return image_details(name, num_dimensions, direction, num_channels,
        kernel_parameters::channel_descriptor_for(element_type_name), required, name_aliases);
}

name_set buffer_names(const kernel_adapter& kernel_adapter, parameter_direction_t direction)
{
    return buffer_names(kernel_adapter,
        [direction](const kernel_adapter::single_parameter_details& spd) {
            return spd.direction == direction;
        } );
}

name_set image_names(const kernel_adapter& kernel_adapter, parameter_direction_t direction)
{
    return buffer_names(kernel_adapter,
        [direction](const kernel_adapter::single_parameter_details& spd) {
            return spd.direction == direction and is_image(spd);
        } );
}

name_set image_names(const kernel_adapter& kernel_adapter)
{
    return buffer_names(kernel_adapter, is_image);
}

marshalled_arguments_type kernel_adapter::marshal_kernel_arguments(const execution_context_t& context) const
{
    auto num_params = parameter_details().size();
    marshalled_arguments_type argument_ptrs_and_maybe_sizes;
    argument_ptrs_and_maybe_sizes.pointers.reserve(num_params);
    argument_ptrs_and_maybe_sizes.sizes.reserve(num_params);

    for(const auto& spd : parameter_details()) {
        if (spd.kind == buffer) {
            kernel_adapters::push_back_buffer(argument_ptrs_and_maybe_sizes, context, spd.direction, spd.name);
        }
        else {
            // it's a scalar
            spd.pusher(argument_ptrs_and_maybe_sizes, context, spd.name);
        }
    }

    if (context.ecosystem == execution_ecosystem_t::cuda) {
        argument_ptrs_and_maybe_sizes.pointers.push_back(nullptr);
        // cuLaunchKernels uses a termination by NULL rather than a length parameter.
        // Note: Remember that sizes is unused in this case
    }
    return argument_ptrs_and_maybe_sizes;
}

namespace kernel_adapters {

void push_back_buffer(
    marshalled_arguments_type& argument_ptrs_and_maybe_sizes,
    const execution_context_t& context,
    parameter_direction_t dir,
    const char* buffer_parameter_name)
{
    const auto& buffer_map =
        [&]() -> device_buffers_map const & {
            switch (dir) {
                case parameter_direction_t::in:
                    return context.buffers.device_side.inputs;
                case parameter_direction_t::inout:
                case parameter_direction_t::out:
                    return context.buffers.device_side.outputs;
                case parameter_direction_t::scratch:
                    return context.buffers.device_side.scratch;
                default:
                    throw std::logic_error("Unexpected direction encountered");
            }
        } ();
    auto const & buffer = buffer_map.at(buffer_parameter_name);
    if (context.ecosystem == execution_ecosystem_t::cuda) {
        argument_ptrs_and_maybe_sizes.pointers.push_back(&(buffer.cuda.data()));
    }
    else { // OpenCL
        argument_ptrs_and_maybe_sizes.sizes.push_back(sizeof(cl_mem));
        // Type punning is fun... not :-()
        if (buffer.opencl.is_image) {
            static_assert(sizeof(cl_mem) == sizeof(cl::Image), "Was expecting cl::Image to only have the size of a cl_mem");
            argument_ptrs_and_maybe_sizes.pointers.push_back(&buffer.opencl.image());
        }
        else {
            static_assert(sizeof(cl_mem) == sizeof(cl::Buffer), "Was expecting cl::Buffer to only have the size of a cl_mem");
            argument_ptrs_and_maybe_sizes.pointers.push_back(&(buffer.opencl.raw));
        }
    }
}

} // namespace kernel_adapters


std::size_t apply_size_calc(const size_calculator_type& calc, const execution_context_t& context)
{
    return calc(
        context.buffers.host_side.inputs,
        context.scalar_input_arguments.typed,
        context.preprocessor_definitions.finalized.valueless,
        context.preprocessor_definitions.finalized.valued,
        context.options.forced_launch_config_components);
}

bool is_input (kernel_adapter::single_parameter_details spd) noexcept     { return is_input(spd.direction);  }
bool is_output(kernel_adapter::single_parameter_details spd) noexcept     { return is_output(spd.direction); }
bool is_buffer(kernel_adapter::single_parameter_details spd) noexcept     { return is_buffer(spd.kind); }
bool is_inout(kernel_adapter::single_parameter_details spd) noexcept      { return spd.direction == parameter_direction_t::inout; }
bool is_scratch(kernel_adapter::single_parameter_details spd) noexcept    { return spd.direction == parameter_direction_t::scratch; }
bool is_scalar(kernel_adapter::single_parameter_details spd) noexcept     { return is_scalar(spd.kind); }
bool is_image(kernel_adapter::single_parameter_details spd)  noexcept     { return is_buffer(spd) and spd.buffer_kind == buffer_kind_t::image; }
bool is_raw_buffer(kernel_adapter::single_parameter_details spd) noexcept { return is_buffer(spd) and spd.buffer_kind == buffer_kind_t::raw; }

// Only invoke this function after having ensured that all necessary information
// about the specified image is present and consistent. Otherwise - it'll throw an exception or terminate
size_t calculate_image_size(
    kernel_adapter::single_parameter_details const& image_buffer_details, execution_context_t const &context)
{
    auto name = image_buffer_details.name;
    // Remember we've already ensured that all necessary information for this argument is present.
    auto dims = util::safe_lookup(context.buffer_dimensions, name);
    if (not dims) {
        die ("No dimensions specified for kernel image parameter '{}'", name);
    }
    if (dims->size() != image_buffer_details.num_dimensions) {
        die ("Invalid number of dimensions specified for kernel image parameter {}: "
             "It is {}-dimensional, but {} dimensions were specified",
            name, image_buffer_details.num_dimensions, dims->size());
    }
    auto num_channels = image_buffer_details.num_channels;
    auto channel_element_type_descriptor = image_buffer_details.element_type;
    // TODO: Trace-print the size calculations here

    auto pitches = util::safe_lookup(context.buffer_pitches, name);
    auto num_pitches = image_buffer_details.num_dimensions - 1;
    if (pitches) {
        if  (pitches->size() != num_pitches) {
            die ("For kernel image parameter {} with {} dimensions, expected {} pitches but {} were specified",
                name, image_buffer_details.num_dimensions, num_pitches, pitches->size());
        }
    }
    auto result = pitches ?
        (*pitches)[num_pitches - 1] * dims->back() :
        channel_element_type_descriptor.size_in_bytes() * num_channels * util::product(*dims);
    return result;
//    spdlog::debug("Kernel argument {}: calculated size is {} bytes", result);
}

optional<size_t> resolve_buffer_size(
    kernel_adapter::single_parameter_details const& buffer_details,
    execution_context_t const& context)
{
    auto cmdline_specified_size = util::safe_lookup(context.output_buffer_sizes, buffer_details.name);
    size_t calculated;
    if (is_image(buffer_details)) {
        calculated = calculate_image_size(buffer_details, context);
    }
    else {
        if (not buffer_details.size_calculator) { return cmdline_specified_size; }
        calculated = buffer_details.size_calculator(
            context.buffers.host_side.inputs,
            context.scalar_input_arguments.typed,
            context.preprocessor_definitions.finalized.valueless,
            context.preprocessor_definitions.finalized.valued,
            context.options.forced_launch_config_components);
    }
    if (cmdline_specified_size and *cmdline_specified_size != calculated) {
        die("Specified size for output buffer {} does not match calculated size: {} != {}",
            buffer_details.name, *cmdline_specified_size, calculated);
    }
    return calculated;
}