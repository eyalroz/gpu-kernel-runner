#ifndef KERNEL_RUNNER_OPENCL_TYPES_HPP_
#define KERNEL_RUNNER_OPENCL_TYPES_HPP_

#define __CL_ENABLE_EXCEPTIONS
DISABLE_WARNING_PUSH
DISABLE_WARNING_IGNORED_ATTRIBUTES
DISABLE_WARNING_SIGN_CONVERSION
#include <khronos/cl2.hpp>
DISABLE_WARNING_POP

#include <chrono>
#include <array>
#include <cassert>

using opencl_duration_type = std::chrono::duration<std::uint64_t, std::nano>;
static_assert(sizeof(std::uint64_t) == sizeof(cl_ulong), "Unexpected size for cl_ulong");

struct raw_opencl_launch_config {
    using array_type = std::array<std::size_t, 3>;
    array_type block_dimensions;
    array_type grid_dimensions; // in blocks!
    // TODO: Support offsets
    // std::size_t offset[3];

    // TODO: dimensions
protected:
    static cl_uint last_nontrivial_dimension_(const std::size_t* dims) {
        assert(dims[0] > 0 and dims[1] > 0 and dims[2] > 0 and "0 dimensions are invalid");
        return (dims[2] > 1) ? 2 : (dims[1] > 0) ? 1 : 0;
    }
    static cl_uint last_nontrivial_dimension_(const array_type& dims) {
        return last_nontrivial_dimension_(dims.data());
    }

    array_type raw_global_dims() const {
        return {
            grid_dimensions[0] * block_dimensions[0],
            grid_dimensions[1] * block_dimensions[1],
            grid_dimensions[2] * block_dimensions[2],
        };
    }

public:

    cl_uint last_nontrivial_dimension() const {
        return last_nontrivial_dimension_(raw_global_dims());
    }

    cl::NDRange offset() const { return cl::NullRange; };

    cl::NDRange global_dims() const {
        auto prod = raw_global_dims();
        switch(last_nontrivial_dimension_(prod)) {
        case 0: return {prod[0]};
        case 1: return {prod[0], prod[1]};
        case 2:
        default: // must be 2
            return {prod[0], prod[1], prod[2]};
        }
    };

    cl::NDRange local_dims() const {
        // Note: We're using the _global_ dimensions - which are no less, but
        // perhaps more, than the _local_ dimensions - since OpenCL doesn't like
        // it otherwise (may trigger a SIGFPE for example).
        switch(last_nontrivial_dimension()) {
        case 0: return {block_dimensions[0]};
        case 1: return {block_dimensions[0], block_dimensions[1]};
        case 2:
        default: // must be 2
            return {block_dimensions[0], block_dimensions[1], block_dimensions[2]};
        }
    };
};


#endif // KERNEL_RUNNER_OPENCL_TYPES_HPP_
