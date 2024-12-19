
#include "kernel_adapter.hpp"

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

kernel_adapter::parameter_details_type kernel_adapter::buffer_details() const
{
    parameter_details_type all_params = parameter_details();
    return util::filter(all_params, [](const single_parameter_details& param) { return param.kind == buffer; });
}

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

kernel_adapter::single_parameter_details kernel_adapter::buffer_details(
    const char*            name,
    parameter_direction_t  direction,
    size_calculator_type   size_calculator,
    bool                   required ,
    std::initializer_list<std::string> name_aliases)
{
    if ((direction == scratch) and (size_calculator == no_size_calc)) {
        throw std::invalid_argument("Scratch buffer parameters must be defined with a size calculator");
    }
    return single_parameter_details {
        name, name_aliases, buffer, no_parser, size_calculator,
        no_pusher, direction, required};
}

name_set buffer_names(const kernel_adapter& kernel_adapter, parameter_direction_t direction)
{
    return buffer_names(kernel_adapter,
        [direction](const kernel_adapter::single_parameter_details& spd) {
            return spd.direction == direction;
        } );
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
    else {
        argument_ptrs_and_maybe_sizes.pointers.push_back(&(buffer.opencl));
        argument_ptrs_and_maybe_sizes.sizes.push_back(sizeof(cl::Buffer));
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

bool is_input (kernel_adapter::single_parameter_details spd) { return is_input(spd.direction);  }
bool is_output(kernel_adapter::single_parameter_details spd) { return is_output(spd.direction); }
