#ifndef KERNEL_RUNNER_OPENCL_BUILD_HPP_
#define KERNEL_RUNNER_OPENCL_BUILD_HPP_

#include <opencl-related/types.hpp>
#include <opencl-related/ugly_error_handling.hpp>
#include <spdlog/spdlog.h>
#include <util/spdlog-extra.hpp>
#include "util/buffer_io.hpp"

#include <string>

// TODO: Use the generated PTX for the device index!
std::string obtain_ptx(const cl::Program &built_program, device_id_t device_id)
{
    auto device_id_ = (size_t) device_id;
    const std::vector<size_t> ptx_sizes = built_program.getInfo<CL_PROGRAM_BINARY_SIZES>();
    if (ptx_sizes.empty()) {
        spdlog::error("No PTXes have been generated for the OpenCL program");
        return {};
    }
    if (ptx_sizes.size() <= device_id_) {
        spdlog::error("No PTX generated for device {}", device_id);
        return {};
    }
//    if (spdlog::level_is_at_least(spdlog::level::debug)) {
//        for(auto i = 0; i < ptx_sizes.size(); i++) {
//            spdlog::debug("PTX for device {} is {}", i, ptx_sizes[i]);
//        }
//    }

    if (ptx_sizes[device_id_] == 0) {
        spdlog::error("Generated PTX for device {} is empty", device_id);
        return {};
    }

    auto ptxes = built_program.getInfo<CL_PROGRAM_BINARIES>();
    if (ptxes.empty()) {
        spdlog::error("No PTX sources are available");
        return {};
    }
    return ptxes[device_id_];
}

std::string marshal_opencl_compilation_options(
    bool                               hopefully_generate_debug_info,
    bool                               generate_source_line_info,
    include_paths_t                    include_paths,
    preprocessor_definitions_t         valueless_definitions,
    valued_preprocessor_definitions_t  valued_definitions,
    const std::vector<std::string>     extra_options)
{
    std::stringstream ss;

//    if (not language_standard.empty()) {
//        ss << " --cl-std=" << language_standard;
//    }

    if (hopefully_generate_debug_info) {
        ss << " -g"; // May not do anything; official Khronos OpenCL semantics rather murky
    }
    if (generate_source_line_info) {
        ss << " -nv-line-info";
    }

    // Should we pre-include anything?
    // Also, are the pre-includes searched in the headers provided to nvrtcCreateProgram,
    // or in the include path? Or both?

    for(const auto& def : valueless_definitions) {
        ss << " -D " << def;
    }

    for(const auto& def_pair : valued_definitions) {
        ss << " -D " << def_pair.first << '=' << def_pair.second;
    }

    // TODO: Perhaps the OpenCL 2.x clCompileProgram mechanism instead?
    // TODO: Check paths for spaces?
    for(const auto& path : include_paths) {
        ss << " -I " << path;
    }

    for(const auto& opt : extra_options) {
        ss << ' ' << opt;
    }

    // TODO: Should we add -cl-kernel-arg-info ?

    if (spdlog::level_is_at_least(spdlog::level::debug)) {
        spdlog::debug("Kernel compilation generated-command-line arguments: \"{}\"", ss.str());
    }
    return ss.str();
}

/*
template <typename InsertIterator>
void add_preincludes(
    InsertIterator, // iterator,
    const include_paths_t&, // include_dir_paths,
    const include_paths_t& // preinclude_files
)
{
    return;
}
*/

std::vector<host_buffer_t>
load_preinclude_files(const include_paths_t& preincludes, const include_paths_t& include_dirs)
{
    // TODO: What about the source file directory?
    auto include_dir_fs_paths = util::transform<std::vector<filesystem::path>>(
        include_dirs, [](const auto& dir) { return filesystem::path{dir}; });
    return util::transform<std::vector<host_buffer_t>>(
        preincludes,
        [&](const auto& preinclude_file_path_suffix) {
            for(const auto& p : include_dir_fs_paths) {
                auto preinclude_path = p / preinclude_file_path_suffix;
                if (filesystem::exists(preinclude_path)) {
                    spdlog::debug("Loading pre-include \"{}\" from {}", preinclude_file_path_suffix, preinclude_path.native());
                    return util::read_file_as_null_terminated_string(preinclude_path);
                }
            }
            spdlog::error("Could not locate preinclude \"{}\"", preinclude_file_path_suffix);
			throw std::invalid_argument("Preinclude loading failed due to missing preinclude"); 
        });
}

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
    std::vector<std::string>         extra_compilation_options)
{
    // TODO: Consider moving the preinclude reading out of this function
    std::vector<host_buffer_t> loaded_preincludes =
        load_preinclude_files(preinclude_files, finalized_include_dir_paths);
    spdlog::debug("All pre-includes loaded from files.");
    auto sources = util::transform<cl::Program::Sources>(loaded_preincludes,
        [](const auto& loaded_file_buffer){
            return std::make_pair(loaded_file_buffer.data(), strlen(loaded_file_buffer.data()));
        });
    sources.push_back(std::make_pair(kernel_source, strlen(kernel_source)));
    cl::Program program = cl::Program(context, sources);
    spdlog::debug("OpenCL program created.");

    std::vector<cl::Device> wrapped_device{device}; // Yes, it's a dumb macro - but it's what OpenCL uses!

    std::string build_options = marshal_opencl_compilation_options(
        generate_debug_info,
        generate_source_line_info,
        finalized_include_dir_paths,
        preprocessor_definitions.valueless,
        preprocessor_definitions.valued,
        extra_compilation_options);

    try {
        program.build(wrapped_device, build_options.c_str());
    } catch(cl::Error& e) {
        cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
        if (status == CL_BUILD_NONE or status == CL_BUILD_IN_PROGRESS) {
            throw std::logic_error("Unexpected OpenCL build status encountered: Expected either success or failure");
        }
        if (status == CL_BUILD_SUCCESS) {
            throw std::logic_error("OpenCL build threw an error, but the build status indicated success");
        }
        std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        auto compilation_failed { false };
        return { compilation_failed, log, {}, {}, nullopt };
    }
    spdlog::trace("OpenCL program built successfully.");
    std::string compilation_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);

    std::string ptx = need_ptx ? obtain_ptx(program, device_id) : std::string{};
    if (need_ptx) {
        spdlog::debug("Got PTX of size {} for target device", ptx.length(), device_id);
    }

    spdlog::debug("Creating OpenCL kernel object for kernel '{}'.", kernel_name);
    try {
        cl::Kernel kernel(program, kernel_name);
        spdlog::trace("OpenCL kernel object created.");
        auto compilation_succeeded { true };
        return { compilation_succeeded, compilation_log, std::move(program), std::move(kernel), std::move(ptx) };
    } catch(cl::Error& ex) {
        spdlog::error("Failed creating kernel; OpenCL error: {}",  clGetErrorString(ex.err()));
        throw ex;
    }
}

#endif // KERNEL_RUNNER_OPENCL_BUILD_HPP_
