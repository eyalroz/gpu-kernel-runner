#ifndef VECTOR_ACCUMULATE_KERNEL_ADAPTER_HPP_
#define VECTOR_ACCUMULATE_KERNEL_ADAPTER_HPP_

#include <kernel_adapter.hpp>

namespace kernel_adapters {

class vector_accumulate final : public kernel_adapter {
public:
    using parent = kernel_adapter;
    using length_type = size_t;

    KA_KERNEL_FUNCTION_NAME("vectorAccumulate" )
    KA_KERNEL_KEY("bundled_with_runner/vector_accumulate" )

    const parameter_details_type& parameter_details() const override
    {
        static const parameter_details_type pd = {
            buffer_details("A", inout).alias("accumulator"),
            buffer_details("B", input),
            scalar_details<length_type>("length").aliases({"size", "num_elements", "nelements"}),
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
            auto length = get_scalar_argument<length_type>(context, "length");
            if (a.size() != length) { return false; }
        }
        return true;
    }

    // Note: We don't have to implement this method - it is merely a convenience
    virtual optional_launch_config_components_t deduce_launch_config(const execution_context_t& context) const override
    {
        optional_launch_config_components_t result;
        result.block_dimensions = std::array<std::size_t,3>{256, 1, 1};
        result.overall_grid_dimensions = std::array<std::size_t,3>{
            any_cast<std::size_t>(context.scalar_input_arguments.typed.at("length")), 1, 1};
        result.dynamic_shared_memory_size = 0;
        return result;
    }

    preprocessor_definition_details_type preprocessor_definition_details() const override
    {
        static const preprocessor_definition_details_type preprocessor_definitions = {
            { "A_LITTLE_EXTRA", isnt_required },
        };
        return preprocessor_definitions;
    }
};

} // namespace kernel_adapters

#endif /* VECTOR_ACCUMULATE_KERNEL_ADAPTER_HPP_ */
