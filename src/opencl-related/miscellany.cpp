#include "miscellany.hpp"
#include "../util/miscellany.hpp"

#include <cuda/api/device.hpp>

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

