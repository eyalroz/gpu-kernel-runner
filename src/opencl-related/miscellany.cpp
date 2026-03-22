#include "miscellany.hpp"
#include "../util/miscellany.hpp"

#include <cuda/api/device.hpp>
#include <spdlog/common.h>

template <>
void ensure_gpu_device_validity_<execution_ecosystem_t::opencl>(
    optional<unsigned> platform_id, int device_id, bool)
{
    constexpr const unsigned OpenCLDefaultPlatformID { 0 };
    auto actual_platform_id = platform_id.value_or(OpenCLDefaultPlatformID);
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (actual_platform_id >= platforms.size()) {
        throw std::invalid_argument("Invalid platform index specified (outside the valid range 0.. " +
            std::to_string(cuda::device::count()-1) + "");
    }
    const auto& platform = platforms[actual_platform_id];
    cl_context_properties properties[] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties) (platform)(),
        0 };
    auto context = cl::Context{CL_DEVICE_TYPE_GPU, properties};
    auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
    if (device_id < 0 or device_id >= (int) devices.size()) {
        throw std::invalid_argument("Invalid device index specified (outside the valid range 0.." +
            std::to_string(cuda::device::count()-1) + ")");
    }
    return;
}

namespace kernel_parameters {

// TODO: Why do we need this hasher, anyways?
struct element_type_descriptor_hasher_t {
    std::size_t operator() (const element_type_descriptor_t& td) const noexcept {
        auto h1 = std::hash<size_t>{}(td.num_bits);
        auto h2 = std::hash<bool>{}(td.signed_);
        auto h3 = std::hash<bool>{}(td.normalized);
        auto h4 = std::hash<int>{}(static_cast<int>(td.numeric_category));
        return h1 ^ h2 ^ h3 ^ h4;
    }
};

bool operator==(const element_type_descriptor_t& lhs, const element_type_descriptor_t& rhs)
{
    return
        lhs.num_bits == rhs.num_bits and
        lhs.signed_ == rhs.signed_ and
        lhs.normalized == rhs.normalized and
        lhs.numeric_category == rhs.numeric_category;
}

} // namespace kernel_parameters

cl_channel_order get_image_channel_order(device_side_buffer_info_t const & buffer_info)
{
    static const cl_channel_order orders[] ={ CL_R, CL_RG, CL_RGB, CL_RGBA };
    if (not util::in_range(buffer_info.num_channels, {1, 4})) {
        throw std::invalid_argument("Unexpected number of image channels " + std::to_string(buffer_info.num_channels));
    }
    return orders[buffer_info.num_channels - 1];
}

cl_channel_type get_image_channel_type(device_side_buffer_info_t const & buffer_info)
{
    using namespace kernel_parameters;
    static const std::unordered_map<kernel_parameters::element_type_descriptor_t, cl_channel_type, kernel_parameters::element_type_descriptor_hasher_t>
    //static const std::pair<element_type_descriptor_t, cl_channel_type>
    descriptors_to_type_code = {
        { {  8, is_signed,   is_normalized,   integral       }, CL_SNORM_INT8 },
        { { 16, is_signed,   is_normalized,   integral       }, CL_SNORM_INT16 },
        { {  8, is_unsigned, is_normalized,   integral       }, CL_UNORM_INT8 },
        { { 16, is_unsigned, is_normalized,   integral       }, CL_UNORM_INT16 },
        { {  8, is_signed,   isnt_normalized, integral       }, CL_SIGNED_INT8 },
        { { 16, is_signed,   isnt_normalized, integral       }, CL_SIGNED_INT16 },
        { { 32, is_signed,   isnt_normalized, integral       }, CL_SIGNED_INT32 },
        { {  8, is_unsigned, isnt_normalized, integral       }, CL_UNSIGNED_INT8 },
        { { 16, is_unsigned, isnt_normalized, integral       }, CL_UNSIGNED_INT16 },
        { { 32, is_unsigned, isnt_normalized, integral       }, CL_UNSIGNED_INT32 },
        { { 16, is_signed,   isnt_normalized, floating_point }, CL_HALF_FLOAT },
        { { 32, is_signed,   isnt_normalized, floating_point }, CL_FLOAT },
    };
    return descriptors_to_type_code.at(buffer_info.channel_elements_type);
}

std::string format_as(const cl::size_t<3>& dimensions)
{
    if (dimensions[1] == 0) { return spdlog::fmt_lib::format("({})", dimensions[0]); }
    if (dimensions[2] == 0) { return spdlog::fmt_lib::format("({},{})", dimensions[0], dimensions[1]); }
    return spdlog::fmt_lib::format("({},{},{})", dimensions[0], dimensions[1], dimensions[2]);
}

