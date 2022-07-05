#ifndef VECTOR_ADD_KERNEL_ADAPTER_HPP_
#define VECTOR_ADD_KERNEL_ADAPTER_HPP_

#include "kernel_adapter.hpp"


namespace kernel_adapters {

class vector_add final : public kernel_adapter {
public:
    using parent = kernel_adapter;
    using length_type = size_t;

    KA_KERNEL_FUNCTION_NAME("vectorAdd")
    KA_KERNEL_KEY("bundled_with_runner/vector_add")

    const parameter_details_type& parameter_details() const override
    {
        static const parameter_details_type pd = {
            // Name      Kind     Parser               Size calculator  Pusher                Direction  Required        Description
            //------------------------------------------------------------------------------------------------------------------------------------------
            {  "C",      buffer,  no_parser,           size_of_A,       no_pusher,            output,    is_required,    "Sequence of sums"             },
            {  "A",      buffer,  no_parser,           no_size_calc,    no_pusher,            input,     is_required,    "First sequence of addends"    },
            {  "B",      buffer,  no_parser,           no_size_calc,    no_pusher,            input,     is_required,    "Second sequence of addends"   },
            {  "length", scalar,  parser<length_type>, no_size_calc,    pusher<length_type>,  input,     is_required,    "Length of each of A, B and C" }
        };
        return pd;
    }

protected:
    static KA_SIZE_CALCULATOR_BY_INPUT_BUFFER(size_of_A, A);

public:
    virtual bool input_sizes_are_valid(const execution_context_t& context) const override
    {
        auto length = get_scalar_argument<length_type>(context, "length");
        const auto& a = context.buffers.host_side.inputs.at("A");
        if (a.size() != length) { return false; }
        const auto& b = context.buffers.host_side.inputs.at("B");
        if (b.size() != length) { return false; }
        return true;
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
