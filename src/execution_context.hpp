#ifndef EXECUTION_CONTEXT_HPP_
#define EXECUTION_CONTEXT_HPP_

#include "common_types.hpp"

#include "kernel_inspecific_cmdline_options.hpp"
#include "launch_configuration.hpp"
#include "preprocessor_definitions.hpp"

#include <util/miscellany.hpp>
#include <util/warning_suppression.hpp>

#include <cuda/api.hpp>
#include <cuda.h> // the CUDA driver API

#define __CL_ENABLE_EXCEPTIONS
DISABLE_WARNING_PUSH
DISABLE_WARNING_IGNORED_ATTRIBUTES
#include <khronos/cl2.hpp>
DISABLE_WARNING_POP
#include <opencl-related/types.hpp>

#include <string>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <memory>
#include <utility>
#include <tuple>

using string_map = std::unordered_map<std::string, std::string>;
using include_paths_t = std::vector<std::string>;
using buffer_sizes = std::unordered_map<std::string, size_t>;

// TODO: Switch to a variant, perhaps?
union device_buffer_type {
    cl::Buffer opencl;
    poor_mans_span cuda;

    // A brittle and dangerous hack
    device_buffer_type() { std::memset(this, 0, sizeof(*this)); }
    device_buffer_type(const device_buffer_type& other) { std::memcpy(this, &other, sizeof(other)); }
    ~device_buffer_type() { }
};

using device_buffers_map = std::unordered_map<std::string, device_buffer_type>;
using scalar_arguments_map = std::unordered_map<std::string, any>;
struct marshalled_arguments_type {
    std::vector<const void*> pointers;
    std::vector<size_t> sizes;
};
class kernel_adapter;

// Essentially, a manually-managed closure and some other dynamically-generated data
struct execution_context_t {
    kernel_inspecific_cmdline_options_t parsed_inspecific_options;
    std::unique_ptr<kernel_adapter> kernel_adapter_;
    struct {
        struct {
            host_buffers_map inputs, outputs; // , expected;
        } host_side;
        struct {
            device_buffers_map inputs, outputs;
                // Note: in-out buffers have one pristine copy in the inputs map,
                // and a "working" copy the outputs map
        } device_side;
        struct {
            string_map inputs, outputs; // , expected;
        } filenames;
    }               buffers;
        // Note: in-out buffers will appear both in the input and the output buffer maps;
        // The input copy will not be used by the kernel directly; rather, before a run,
        // it will first be copied to the output, and the "output" buffer will be used
        // instead.
    device_id_t     device_id;
    execution_ecosystem_t ecosystem;
    struct {
        int             driver_device_id;
        // With the CUDA driver, there are two kinds/two levels of device IDs. it's a bit
        // defat but that's just the way it is.
        CUcontext       context;
        CUfunction      built_kernel;
        CUmodule        module;
    } cuda;
    struct {
        cl::Context       context;
        cl::Device        device;
        cl::Program       program;
        cl::Kernel        built_kernel;
        cl::CommandQueue  queue;
        std::vector<std::size_t> finalized_argument_sizes;
            // TODO: Consider moving these out of the OpenCL-specific structure
    } opencl;
    std::string     compiled_ptx; // PTX or whatever OpenCL become.
    struct {
        string_map raw; // the strings passed on the command-line for the arguments
        scalar_arguments_map typed; // the parsed values for each scalar, after type-erasure
    } scalar_input_arguments;
    // We use this next field since we get some preprocessor definitions through
    // specific command-line options
    preprocessor_value_definitions_t  option_specified_preprocessor_definitions;
    std::vector<std::string>
                    parameter_names; // for determining how to pass the arguments
    struct {
        preprocessor_definitions_t valueless;
        preprocessor_value_definitions_t valued;
    } finalized_preprocessor_definitions;
    include_paths_t finalized_include_dir_paths;
    marshalled_arguments_type finalized_arguments;
    launch_configuration_type kernel_launch_configuration;

public:

    template <typename T>
    T get_defined_value(const std::string& s)
    {
        return ::get_defined_value<T>(finalized_preprocessor_definitions.valued, s);
    }
};

// TODO: This may not be such a good idea, rethink it.

template <>
inline bool execution_context_t::get_defined_value<bool>(const std::string& s)
{
    auto find_result = finalized_preprocessor_definitions.valueless.find(s);
    if (find_result != std::cend(finalized_preprocessor_definitions.valueless)) {
        return true;
    }
    return ::get_defined_value<bool>(finalized_preprocessor_definitions.valued, s);
}

#endif /* EXECUTION_CONTEXT_HPP_ */
