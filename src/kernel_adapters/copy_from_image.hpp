#ifndef COPY_FROM_IMAGE_KERNEL_ADAPTER_HPP_
#define COPY_FROM_IMAGE_KERNEL_ADAPTER_HPP_

#include <kernel_adapter.hpp>

namespace kernel_adapters {

class copy_from_image final : public kernel_adapter {
public:
    using length_type = size_t;

    KA_KERNEL_FUNCTION_NAME("copyFromImage")
    KA_KERNEL_KEY("bundled_with_runner/copy_from_image")

    const parameter_details_type& parameter_details() const override
    {
        static const parameter_details_type pd = {
            image_details("source", 2_dims, input, 1_channels, "uchar", is_required, { "image", "src" }),
            raw_buffer_details("destination", output, no_size_calc, is_required, { "output", "out", "dest" })
        };
        return pd;
    }
};

} // namespace kernel_adapters

#endif /* COPY_FROM_IMAGE_KERNEL_ADAPTER_HPP_ */
