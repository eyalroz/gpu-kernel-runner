#ifndef VECTOR_ADD_KERNEL_ADAPTER_HPP_
#define VECTOR_ADD_KERNEL_ADAPTER_HPP_


#include "kernel_adapter.hpp"
#include "util/optional_and_any.hpp"
#include <cstdlib>
#include <array>

// TODO:
// 1. To make adapters simpler to write and avoid DRY,
// we should have a single descriptor structure for all parameters, in order,
// which, for each of them, indicates its direction, whether it's a scalar or buffer,
// perhaps its constness, its name and its description. By filtering and projecting this
// structure, we can obtain various structures we return or iterate-over in different
// methods of a kernel adapter class.
// 2. Similar arrangement for preprocessor definitions: With/without value, optional/required,
//    name/term and description.
//
// Doing the above should allow us to stick to base-class implementations of:
//
//   input_buffer_names()
//   output_buffer_names()
//   cmdline_required_preprocessor_definition_terms()
//   output_buffer_sizes()
//   input_buffer_details()
//   output_buffer_details()
//   add_scalar_arguments_cmdline_options()
//   marshal_kernel_arguments_inner()
//
// and perhaps even parse_cmdline_scalar_argument

namespace kernel_adapters {

class vector_add final : public kernel_adapter {
public:
    using parent = kernel_adapter;
    using length_type = unsigned int;

    constexpr static const char* kernel_function_name_ { "vectorAdd" };
    constexpr static const char* key_ { "bundled_with_runner/vector_add" };

    vector_add() : kernel_adapter() { }
    vector_add(const vector_add& other) = default;
    ~vector_add() = default;

    std::string kernel_function_name() const override { return kernel_function_name_; }
    std::string key() const override { return key_; }

    const buffer_details_type& buffer_details() const override
    {
        constexpr const auto input  = parameter_direction_t::input;
        constexpr const auto output = parameter_direction_t::output;
        constexpr const auto inout  = parameter_direction_t::inout;

        static const buffer_details_type buffer_details_ = {
            { "A", input,  "First sequence of addends"},
            { "B", input,  "Second sequence of addends"},
            { "C", output, "Sequence of two-element sums"},
        };
        return buffer_details_;
    }

    const scalar_details_type& scalar_argument_details() const override
    {
        static const scalar_details_type scalar_argument_details_ = {
        	{"length", "Length of each of A, B and C" }
        };
        return scalar_argument_details_;
    }

    parameter_name_set cmdline_required_scalar_argument_names() const override {
        return {};
    }

    parameter_name_set cmdline_required_preprocessor_definition_terms() const override
    {
        return { "A_LITTLE_EXTRA" };
    }

    scalar_arguments_map generate_additional_scalar_arguments(execution_context_t& context) const override
    {
        const auto& A = context.buffers.host_side.inputs.at("A");
        scalar_arguments_map generated;
        generated["length"] = any(static_cast<length_type>(A.size()));
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
        const scalar_arguments_map& scalar_arguments,
        const preprocessor_definitions_t& valueless_definitions,
        const preprocessor_value_definitions_t& valued_definitions) const override
    {
        return {
            { "C", input_buffers.at("A").size() }
        };
    }

    bool extra_validity_checks(const execution_context_t& context) const override
    {
        return true; // Our checks are covered elsewhere
    }

    virtual bool input_sizes_are_valid(const execution_context_t& context) const override
    {
        const auto& A = context.buffers.host_side.inputs.at("A");
        const auto& B = context.buffers.host_side.inputs.at("B");
        if (A.size() != B.size()) {
            return false;
        }
        // TODO: Implement a contains() function
        if (context.scalar_input_arguments.typed.find("length") !=
            context.scalar_input_arguments.typed.cend())
        {
            const auto& length_any = context.scalar_input_arguments.typed.at("length");
            auto length = any_cast<length_type>(length_any);
            if (A.size() != length) { return false; }
        }
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

    virtual optional_launch_config_components deduce_launch_config(const execution_context_t& context) const override
    {
        optional_launch_config_components result;
        auto length = any_cast<length_type>(context.scalar_input_arguments.typed.at("length"));
        result.block_dimensions = std::array<std::size_t,3>{256, 1, 1};
        result.overall_grid_dimensions = std::array<std::size_t,3>{length, 1, 1};
        result.dynamic_shared_memory_size = 0;
        return result;
    }

    virtual const preprocessor_definitions_type& preprocessor_definition_details() const override
    {
        constexpr const bool is_required = single_preprocessor_definition_details::is_required;

        static const preprocessor_definitions_type preprocessor_definitions = {
            { "A_LITTLE_EXTRA", "Something extra to add to the result", is_required }
        };
        return preprocessor_definitions;
    }
};

} // namespace kernel_adapters

#endif /* VECTOR_ADD_KERNEL_ADAPTER_HPP_ */
