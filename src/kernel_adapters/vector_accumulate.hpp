#ifndef VECTOR_ACCUMULATE_KERNEL_ADAPTER_HPP_
#define VECTOR_ACCUMULATE_KERNEL_ADAPTER_HPP_

#include "kernel_adapter.hpp"

namespace kernel_adapters {

class vector_accumulate final : public kernel_adapter {
public:
    using parent = kernel_adapter;
    using length_type = size_t;

    constexpr static const char* kernel_function_name_ { "vectorAccumulate" };
    constexpr static const char* key_ { "bundled_with_runner/vector_accumulate" };

    std::string kernel_function_name() const override { return kernel_function_name_; }
    std::string key() const override { return key_; }

    const parameter_details_type& parameter_details() const override
    {
        static const parameter_details_type pd = {
            // Name      Kind     Direction  Required       Description
            //------------------------------------------------------------------------------
            {  "A",      buffer,  inout,     is_required,   "Accumulator sequence (initialized with a second sequence of addends)" },
            {  "B",      buffer,  input,     is_required,   "First sequence of addends" },
            {  "length", scalar,  input,     isnt_required, "Length of each of A and B" }
        };
        return pd;
    }

    // Note: If we marked "length" as being required, we would not need to implement this
    scalar_arguments_map generate_additional_scalar_arguments(execution_context_t& context) const override
    {
        const auto& a = context.buffers.host_side.inputs.at("A");
        scalar_arguments_map generated;
        generated["length"] = any(a.size());
        return generated;
    }

    any parse_cmdline_scalar_argument(const std::string& argument_name, const std::string& argument) const override {
        if (argument_name == "length") {
            return { static_cast<length_type>(std::stoull(argument)) };
        }
        throw std::invalid_argument("No scalar argument " + argument_name);
    }

    // Note: The actual size might be smaller; this is what we need to allocate
    buffer_sizes output_buffer_sizes(
        const host_buffers_map& input_buffers,
        const scalar_arguments_map&,
        const preprocessor_definitions_t&,
        const preprocessor_value_definitions_t&) const override
    {
        return {
            { "A", input_buffers.at("A").size() }
        };
    }

    virtual bool input_sizes_are_valid(const execution_context_t& context) const override
    {
        const auto& a = context.buffers.host_side.inputs.at("A");
        const auto& b = context.buffers.host_side.inputs.at("B");
        if (a.size() != b.size()) {
            return false;
        }
        // Note: If we marked "length" as being required, we would not need to check this
        if (context.scalar_input_arguments.typed.find("length") !=
            context.scalar_input_arguments.typed.cend())
        {
            const auto& length_any = context.scalar_input_arguments.typed.at("length");
            auto length = any_cast<length_type>(length_any);
            if (a.size() != length) { return false; }
        }
        return true;
    }

    void marshal_kernel_arguments_inner(
        marshalled_arguments_type& args,
        const execution_context_t& context) const override
    {
        push_back_buffer(args, context, parameter_direction_t::inout, "A");
        push_back_buffer(args, context, parameter_direction_t::input, "B");
        push_back_scalar<length_type>(args, context, "length");
    }

    // Note: We don't have to implement this method - it is merely a convenience
    virtual optional_launch_config_components deduce_launch_config(const execution_context_t& context) const override
    {
        optional_launch_config_components result;
        result.block_dimensions = std::array<std::size_t,3>{256, 1, 1};
        result.overall_grid_dimensions = std::array<std::size_t,3>{
            any_cast<std::size_t>(context.scalar_input_arguments.typed.at("length")), 1, 1};
        result.dynamic_shared_memory_size = 0;
        return result;
    }

    virtual const preprocessor_definitions_type& preprocessor_definition_details() const override
    {
        static const preprocessor_definitions_type preprocessor_definitions = {
            { "A_LITTLE_EXTRA", "Something extra to add to the result", not is_required }
        };
        return preprocessor_definitions;
    }
};

} // namespace kernel_adapters

#endif /* VECTOR_ACCUMULATE_KERNEL_ADAPTER_HPP_ */
