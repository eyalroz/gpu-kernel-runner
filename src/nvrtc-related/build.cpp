/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2020, GE Healthcare.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "build.hpp"

#include "../util/filesystem.hpp"

using include_dir_paths_t = std::vector<std::string>;

bool check_potential_cuda_include_dir(filesystem::path candidate_dir)
{
    return (filesystem::exists(candidate_dir) and has_permission(candidate_dir, for_recursion));
}

// This is a stub. It's not terribly hard to do, but it's more than a few lines
// of code - certainly if you want it to work on MacOs Windows as well as Linux.
optional<std::string> locate_cuda_include_directory()
{
    filesystem::path candidate;

    constexpr const char *cuda_root_env_var_names[] = { "CUDA_ROOT", "CUDA_PATH", "CUDA_DIR" };
    for (auto env_var_name : cuda_root_env_var_names) {
        auto cuda_root_env_dir = util::get_env(env_var_name);
        if (cuda_root_env_dir) {
            candidate = filesystem::path(cuda_root_env_dir.value()) / "include";
            if (check_potential_cuda_include_dir(candidate)) {
                spdlog::trace( "Using the CUDA include directory specified in {}, "
                   "to append to  the compilation options", env_var_name);
                return candidate.native();
            }
        }
    }

#ifdef CUDA_INCLUDE_DIR
    candidate = CUDA_INCLUDE_DIR;
    if (check_potential_cuda_include_dir(candidate)) {
        return candidate.native();
    }
#endif
    // TODO: Check the PATH for the CUDA binaries dir
    return nullopt;
}

/**
 * We let the user specify an arbitrary set of "extra" compilation options; however, some of
 * these might clahs with, or duplicate, options our code itself sets. NVRTC (as of CUDA 12.6)
 * is rather picky about these situations, even if one is jus trepliacing the same switch more
 * than once. Ideally, the options class itself should contain code which removes duplicates
 * from the extra options and identifes clashes, but - it doesn't, for now, so let's do something
 * quick-and-dirty with a few options we know are problematic for NVRTC.
 *
 * @param options The NVRTC compilation option - the ones we set in "proper" fields and
 * additional options, most/all specified by the user
 */
void opportunistically_remove_duplicates(cuda::rtc::compilation_options_t<cuda::cuda_cpp>& options)
{
    auto& extra = options.extra_options;
	spdlog::trace("Before duplicate removal, extra args are: {}", extra);
    if (options.default_execution_space_is_device) {
        util::remove(extra, "--device-as-default-execution-space");
		spdlog::debug("Removed redundnat command-line arg {} ", "--device-as-default-execution-space");
    }

    // Note: assuming the dialect spec using the same arg, e.g. "--std=c++14", rather than a separate
    // one, "--std c++14"

    if (options.language_dialect) {
        const char dialect_param_prefix[] = "--std=";
        auto it = util::find_if(extra, [&](const std::string& s) {
            return s.substr(0, strlen(dialect_param_prefix)) == dialect_param_prefix;
        } );
        if (it != extra.end()) {
            auto main_opts_cpp_dialect_name = cuda::rtc::detail_::cpp_dialect_names[static_cast<int>(*options.language_dialect)];
            auto extra_opts_dialect_name = it->substr(std::strlen(dialect_param_prefix));
            if (main_opts_cpp_dialect_name != extra_opts_dialect_name) {
                options.set_language_dialect(extra_opts_dialect_name);
            }
            extra.erase(it);
        }
    }
    auto it = util::find(extra, "--device-as-default-execution-space" );
    if (it != extra.end()) {
        options.default_execution_space_is_device = true;
        extra.erase(it);
    }
	spdlog::trace("After duplicate removal, extra args are: {}", extra);
}

compilation_result_t build_cuda_kernel(
    const cuda::context_t& context,
    const char* kernel_source_file_path,
    const char* kernel_source,
    const char* kernel_function_name,
    bool set_default_compilation_options,
    bool generate_debug_info,
    bool generate_source_line_info,
    optional<const std::string> language_standard,
    const std::vector<std::string>& include_dir_paths,
    const std::vector<std::string>& preinclude_files,
    const split_preprocessor_definitions_t& preprocessor_definitions,
    const std::vector<std::string>& extra_compilation_options)
{
    auto nvrtc_version = cuda::version_numbers::nvrtc();
    spdlog::debug("Preparing to build the kernel using CUDA's nvrtc JIT compilation, version {}.{} ...",
        nvrtc_version.major, nvrtc_version.minor);
    // TODO: Consider mentioning the kernel function name in the program name.
    auto program = cuda::rtc::program::create<cuda::cuda_cpp>(kernel_source_file_path)
        .set_source(kernel_source)
        .set_headers(get_standard_header_substitutes());

    cuda::rtc::compilation_options_t<cuda::cuda_cpp> opts;

    if (language_standard) {
        opts.set_language_dialect(language_standard.value());
    }
    opts.generate_debug_info = generate_debug_info;
    opts.generate_source_line_info = generate_source_line_info;
    if (set_default_compilation_options) {
        opts.default_execution_space_is_device = true;
        opts.set_target(context.device());
    }
    else {
        spdlog::debug("As requested, NOT setting a compilation option describing the target compute capability");
    }
    // TODO: Note the copying of strings and maps here; can we move all of these instead?
    opts.additional_include_paths = include_dir_paths;
    opts.preinclude_files = preinclude_files;
    opts.no_value_defines = preprocessor_definitions.valueless;
    opts.valued_defines = preprocessor_definitions.valued;
    opts.extra_options = extra_compilation_options;

    // Ugly kludge :-(
    opportunistically_remove_duplicates(opts);

    spdlog::debug("Kernel compilation generated-command-line arguments: \"{}\"", render(opts));

    program.add_registered_global(kernel_function_name).set_options(opts);

    auto compilation_output = program.compile();
    auto raw_log = compilation_output.log();
    std::string log { raw_log.data(), raw_log.size() };
    if (not compilation_output.succeeded()) {
        constexpr const bool compilation_failed { false };
        return { compilation_failed, std::move(log), nullopt, nullopt, nullopt };
    }
    spdlog::info("Kernel source compiled successfully.");
    bool compilation_succeeded { true };
    std::string mangled_kernel_function_signature = compilation_output.get_mangling_of(kernel_function_name);
    spdlog::trace("Mangled kernel function signature is: {}", mangled_kernel_function_signature);

    compilation_output.has_ptx() or die("No PTX in compiled kernel CUDA program");
    auto ptx = compilation_output.ptx();
    // Yes, it's a copy, the API kind of sucks here
    std::string ptx_as_string = std::string(ptx.data(), ptx.size());
    spdlog::debug("Compiled PTX length: {} characters.", ptx.size());
    auto module = cuda::module::create(context, compilation_output);
    auto kernel = module.get_kernel(mangled_kernel_function_signature);
    spdlog::debug("Kernel static memory usage: {} bytes.",
        kernel.get_attribute(cuda::kernel::attribute_t::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES));
    spdlog::debug("Compiled kernel loaded into a CUDA module.");
    return {
        compilation_succeeded,
        std::move(log),
        std::move(module),
        std::move(kernel),
        std::move(ptx_as_string)
    };
}
