#ifndef VECTOR_ADD_KERNEL_ADAPTER_HPP_
#define VECTOR_ADD_KERNEL_ADAPTER_HPP_

#include "kernel_adapter.hpp"

namespace kernel_adapters {

class vector_add final : public kernel_adapter {
public:
    using parent = kernel_adapter;
    using length_type = size_t;

    constexpr static const char* kernel_function_name_ { "vectorAdd" };
    constexpr static const char* key_ { "bundled_with_runner/vector_add" };

    std::string kernel_function_name() const override { return kernel_function_name_; }
    std::string key() const override { return key_; }

    const parameter_details_type& parameter_details() const override
    {
        static const parameter_details_type pd = {
            // Name      Kind     Direction  Required       Description
            //----------------------------------------------------------------------------
            {  "C",      buffer,  output,    is_required,   "Sequence of sums"             },
            {  "A",      buffer,  input,     is_required,   "First sequence of addends"    },
            {  "B",      buffer,  input,     is_required,   "Second sequence of addends"   },
            {  "length", scalar,  input,     is_required,   "Length of each of A, B and C" }
        };
        return pd;
    }

    any parse_cmdline_scalar_argument(const std::string& parameter_name, const std::string& argument) const override {
        if (parameter_name == "length") {
            return { static_cast<length_type>(std::stoull(argument)) };
        }
        throw std::invalid_argument("No scalar argument " + parameter_name);
    }

    buffer_sizes output_buffer_sizes(
        const host_buffers_map& input_buffers,
        const scalar_arguments_map&,
        const preprocessor_definitions_t&,
        const preprocessor_value_definitions_t&) const override
    {
        return {
            { "C", input_buffers.at("A").size() }
        };
    }

    virtual bool input_sizes_are_valid(const execution_context_t& context) const override
    {
        const auto& length_any = context.scalar_input_arguments.typed.at("length");
        auto length = any_cast<length_type>(length_any);
        const auto& a = context.buffers.host_side.inputs.at("A");
        if (a.size() != length) { return false; }
        const auto& b = context.buffers.host_side.inputs.at("B");
        if (b.size() != length) { return false; }
        return true;
    }

    void marshal_kernel_arguments_inner(
        marshalled_arguments_type& args,
        const execution_context_t& context) const override
    {
        push_back_buffer(args, context, parameter_direction_t::out, "C");
        push_back_buffer(args, context, parameter_direction_t::in,  "A");
        push_back_buffer(args, context, parameter_direction_t::in,  "B");
        push_back_scalar<length_type>(args, context, "length");
    }

    virtual const preprocessor_definitions_type& preprocessor_definition_details() const override
    {
        static const preprocessor_definitions_type preprocessor_definitions = {
            { "A_LITTLE_EXTRA", "Something extra to add to the result", is_required }
        };
        return preprocessor_definitions;
    }
};

} // namespace kernel_adapters

#endif /* VECTOR_ADD_KERNEL_ADAPTER_HPP_ */
