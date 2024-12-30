#ifndef KERNEL_RUNNER_OPENCL_BUILD_HPP_
#define KERNEL_RUNNER_OPENCL_BUILD_HPP_

#include "types.hpp"
#include "ugly_error_handling.hpp"
#include "common_types.hpp"

#include "../preprocessor_definitions.hpp"
#include "../util/buffer_io.hpp"
#include "../util/spdlog-extra.hpp"

#include <spdlog/spdlog.h>

#include <string>

// TODO: Use the generated PTX for the device index!
std::string obtain_ptx(const cl::Program &built_program, device_id_t device_id);

std::string marshal_opencl_compilation_options(
    bool                               hopefully_generate_debug_info,
    bool                               generate_source_line_info,
    include_paths_t                    include_paths,
    preprocessor_definitions_t         valueless_definitions,
    valued_preprocessor_definitions_t  valued_definitions,
    const std::vector<std::string>     extra_options);

std::vector<host_buffer_t>
load_preinclude_files(const include_paths_t& preincludes, const include_paths_t& include_dirs);

struct opencl_compilation_result_t {
    bool succeeded;
    optional<std::string> log;
    cl::Program program; // Don't need this to be optional, since it's not a RAII type
    cl::Kernel kernel; // Don't need this to be optional, since it's not a RAII type
    optional<std::string> ptx;
};

opencl_compilation_result_t build_opencl_kernel(
    cl::Context context,
    cl::Device  device,
    device_id_t device_id,
    const char* kernel_name,
    const char* kernel_source,
    bool        generate_debug_info,
    bool        generate_source_line_info,
    bool        need_ptx,
    const include_paths_t& finalized_include_dir_paths,
    const include_paths_t& preinclude_files,
    split_preprocessor_definitions_t preprocessor_definitions,
    std::vector<std::string>         extra_compilation_options);

#endif // KERNEL_RUNNER_OPENCL_BUILD_HPP_
