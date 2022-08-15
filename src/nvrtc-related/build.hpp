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

#ifndef KERNEL_RUNNER_NVRTC_WRAPPER_HPP_
#define KERNEL_RUNNER_NVRTC_WRAPPER_HPP_

#include "standard_header_substitutes.hpp"

#include <cuda/nvrtc.hpp>

#include <util/miscellany.hpp>
#include <util/spdlog-extra.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/cfg/helpers.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/cfg/env.h>

#include <cuda.h>
#include <nvrtc.h>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <memory>
#include <algorithm>
#include <cstring>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using preprocessor_value_definitions_t = std::unordered_map<std::string, std::string>;
using preprocessor_definitions_t = std::unordered_set<std::string>;
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
#ifdef CUDA_INCLUDE_DIR
    candidate = CUDA_INCLUDE_DIR;
    if (check_potential_cuda_include_dir(candidate)) {
        return candidate.native();
    }
#endif
    constexpr const char *cuda_root_env_var_names[] = { "CUDA_ROOT", "CUDA_PATH", "CUDA_DIR" };
    for (auto env_var_name : cuda_root_env_var_names) {
        auto cuda_root_env_dir = util::get_env(env_var_name);
        if (cuda_root_env_dir) {
            candidate = filesystem::path(cuda_root_env_dir.value()) / "include";
            if (check_potential_cuda_include_dir(candidate)) {
                return candidate.native();
            }
        }
    }
    // TODO: Check the PATH for the CUDA binaries dir
    return nullopt;
}

struct compilation_result_t {
    bool succeeded;
    optional<std::string> log;
    optional<cuda::module_t> module;
    optional<std::string> ptx;
    optional<std::string> mangled_signature;
};

compilation_result_t build_cuda_kernel(
    const cuda::context_t& context,
    const char* kernel_source_file_path,
    const char* kernel_source,
    const char* kernel_function_name,
    bool debug_mode,
    bool generate_line_info,
    const std::string& language_standard,
    const std::vector<std::string>& include_dir_paths,
    const std::vector<std::string>& preinclude_files,
    const preprocessor_definitions_t& preprocessor_definitions,
    const preprocessor_value_definitions_t& preprocessor_value_definitions,
    const std::vector<std::string>& extra_compilation_options)
{
    // TODO: Consider mentioning the kernel function name in the program name.
    auto program = cuda::rtc::program::create(kernel_source_file_path)
        .set_source(kernel_source)
        .set_headers(get_standard_header_substitutes());

    cuda::rtc::compilation_options_t opts;
    opts.set_language_dialect(language_standard);
    opts.debug = debug_mode;
    opts.generate_line_info = generate_line_info;
    // TODO: Note the copying of strings and maps here; can we move all of these instead?
    opts.additional_include_paths = include_dir_paths;
    opts.preinclude_files = preinclude_files;
    opts.no_value_defines = preprocessor_definitions;
    opts.valued_defines = preprocessor_value_definitions;
    opts.default_execution_space_is_device = true;
    opts.set_target(context.device());
    opts.extra_options = extra_compilation_options;

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

    if (not compilation_output.has_ptx()) {
        throw std::runtime_error("No PTX in compiled kernel CUDA program");
    }
    auto ptx = compilation_output.ptx();
    // Yes, it's a copy, the API kind of sucks here
    std::string ptx_as_string = std::string(ptx.data(), ptx.size());
    spdlog::debug("Compiled PTX length: {} characters.", ptx.size());
    auto module = cuda::module::create(context, compilation_output);
    spdlog::debug("Compiled kernel loaded as a CUDA module.");
    return {
        compilation_succeeded,
        std::move(log),
        std::move(module),
        std::move(ptx_as_string),
        std::move(mangled_kernel_function_signature)
    };
}

#endif  // KERNEL_RUNNER_NVRTC_WRAPPER_HPP_
