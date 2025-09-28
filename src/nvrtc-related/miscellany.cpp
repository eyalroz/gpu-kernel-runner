#include "miscellany.hpp"
#include <cuda/api.hpp>
#include "../util/spdlog-extra.hpp"

template <>
std::vector<filesystem::path> get_ecosystem_include_paths_<execution_ecosystem_t::cuda>()
{
    std::vector<filesystem::path> result;
    auto cuda_include_dir = locate_cuda_include_directory();
    if (cuda_include_dir) {
        spdlog::debug("Using CUDA include directory {}", cuda_include_dir.value());
        result.emplace_back(cuda_include_dir.value());
    }
    else {
        spdlog::warn("Cannot locate CUDA include directory - trying to build the kernel with it missing.");
    }
    return result;
}

struct shared_memory_sizes {
    cuda::memory::shared::size_t static_, dynamic_;
    cuda::memory::shared::size_t overall() { return static_ + dynamic_; }
};
/*
size_t determine_required_shared_memory_size(execution_context_t const& context)
{
    if (not context.cuda.kernel) {
        throw std::invalid_argument("Execution context has no built CUDA kernel");
    }
    shared_memory_sizes result;
    result.static_ = context.cuda.kernel->get_attribute(cuda::kernel::attribute_t::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
    result.dynamic_ = context.kernel_launch_configuration.cuda.dynamic_shared_memory_size;
    return result.overall();
}

// This assumes all relevant checks have been performed
void enable_sufficient_shared_memory_inner(
    cuda::kernel_t const& kernel,
    cuda::memory::shared::size_t required_by_kernel,
    cuda::memory::shared::size_t device_max)
{
    if (required_by_kernel == 0) {
        spdlog::trace("Kernel requires {} bytes of shared memory overall, so - not setting any carveout hint.",required_by_kernel);
        return;
    }
    auto required_carveout = util::div_rounding_up(required_by_kernel * 100, device_max);
    spdlog::debug("Kernel requires {} bytes of shared memory overall, device offers {}; "
                  "setting kernel carve-out hint to {}.", required_by_kernel, device_max, required_carveout);
    kernel.set_attribute(CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, required_carveout);
}

void enable_sufficient_shared_memory(execution_context_t const& context, size_t required_shmem_size)
{
    if (not context.cuda.kernel) {
        throw std::invalid_argument("Execution context has no built CUDA kernel");
    }
    auto device = cuda::device::get(context.device_id);
    auto max_available = device.compute_capability().max_shared_memory_per_block();
    if (max_available < required_shmem_size) {
        std::ostringstream  oss;
        oss << "The kernel requires " << required_shmem_size << " bytes of shared memory overall, "
            << "but GPU device " << device.id() << " only supports up to  " << max_available << " bytes of memory per block.";
        throw std::runtime_error(oss.str());
    }
    enable_sufficient_shared_memory_inner(
        *context.cuda.kernel,
        static_cast<cuda::memory::shared::size_t>(required_shmem_size),
        max_available);

}

void enable_sufficient_shared_memory(execution_context_t const& context)
{
    auto required_shmem_size = determine_required_shared_memory_size(context);
    enable_sufficient_shared_memory(context, required_shmem_size);
}
*/

template <>
void ensure_gpu_device_validity_<execution_ecosystem_t::cuda>(
    optional<unsigned>, int device_id, bool)
{
    auto device_count = (std::size_t) cuda::device::count();
    if (device_count == 0) die("No CUDA devices detected on this system");

    // TODO: Move this logic upwards... by using a function which gets the device count
    if (device_id < 0 or device_id >= (int) device_count)
        die ("Invalid device index specified (outside the valid range 0.. {})", cuda::device::count()-1);
}

