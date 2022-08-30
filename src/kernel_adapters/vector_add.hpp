#ifndef VECTOR_ADD_KERNEL_ADAPTER_HPP_
#define VECTOR_ADD_KERNEL_ADAPTER_HPP_

#include <kernel_adapter.hpp>

namespace kernel_adapters {

class vector_add final : public kernel_adapter {
public:
    using length_type = size_t;

    KA_KERNEL_FUNCTION_NAME("vectorAdd")
    KA_KERNEL_KEY("bundled_with_runner/vector_add")

    const parameter_details_type& parameter_details() const override
    {
        static const parameter_details_type pd = {
            // buffer_details("C",  output,    size_by_length),
            buffer_details("C",  output),
            buffer_details("A",  input),
            buffer_details("B",  input),
            scalar_details<length_type>("length"),
        };
        return pd;
    }

// The following are not necessary, but could help this adapter:

// protected:
//    static std::size_t size_by_length(
//        const host_buffers_t&,
//        const scalar_arguments_map& scalars,
//        const preprocessor_definitions_t&,
//        const preprocessor_value_definitions_t&,
//        const optional_launch_config_components_t&)
//    {
//        return any_cast<length_type>(scalars.at("length"));
//    }
//
// public:
//    bool input_sizes_are_valid(const execution_context_t& context) const override
//    {
//        auto length = get_scalar_argument<length_type>(context, "length");
//        const auto& a = context.buffers.host_side.inputs.at("A");
//        if (a.size() != length) { return false; }
//        const auto& b = context.buffers.host_side.inputs.at("B");
//        if (b.size() != length) { return false; }
//        return true;
//    }
//
//    optional<preprocessor_definitions_type> preprocessor_definition_details() const override
//    {
//        static const preprocessor_definitions_type preprocessor_definitions = {
//            { "A_LITTLE_EXTRA", is_required }
//        };
//        return preprocessor_definitions;
//    }
};

} // namespace kernel_adapters

#endif /* VECTOR_ADD_KERNEL_ADAPTER_HPP_ */
