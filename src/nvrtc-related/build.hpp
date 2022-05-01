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

#include "ugly_error_handling.hpp"
#include "standard_header_substitutes.hpp"

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
#include <sstream>
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

struct compilation_result_t {
  nvrtcProgram program;
  std::string ptx;
};


void maybe_print_compilation_log(bool compilation_failed, const nvrtcProgram& nvrtc_program)
{
    // dump log
    size_t logSize;
    cuda_api_call(nvrtcGetProgramLogSize, nvrtc_program, &logSize);

    bool print_the_log = compilation_failed or (logSize >= 2);
    if (not print_the_log) { return; }

    auto compilation_log = std::make_unique<char[]>(logSize + 1);
    cuda_api_call(nvrtcGetProgramLog, nvrtc_program, compilation_log.get());
    compilation_log[logSize] = '\x0';

    spdlog::level::level_enum level = compilation_failed ? spdlog::level::err : spdlog::level::debug ;

    if (logSize < 2 and print_the_log) {
        spdlog::debug("Compilation log is empty.");
    }
    spdlog::log(level, "Compilation log:\n"
        "-----\n"
        "{}"
        "-----", compilation_log.get());
}

std::vector<std::string> marshal_compilation_options(
    CUdevice cuDevice,
    bool compile_in_debug_mode,
    bool generate_line_info,
    const std::string& language_standard,
    const include_dir_paths_t& include_dir_paths,
    const include_dir_paths_t& preinclude_files,
    const preprocessor_definitions_t& preprocessor_definitions,
        // each string has the form "NAME=value" or "NAME" without an equality sign
    const preprocessor_value_definitions_t& preprocessor_definition_pairs
    // Note: Could potentially take more compilation options here. But then - this
    //  is a super-ugly function, I'd rather delete it, not extend it
    )
{
    // --include-path=<dir>
    std::vector<std::string> compilation_cmdline_args;

    int major = 0, minor = 0;

    // get compute capabilities and the devicename
    cuda_api_call(cuDeviceGetAttribute, &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice);
    cuda_api_call(cuDeviceGetAttribute, &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice);

    std::stringstream ss;


    ss << "--gpu-architecture=compute_" << major << minor; // e.g. --gpu-architecture=compute_61
    compilation_cmdline_args.push_back(ss.str());

    compilation_cmdline_args.push_back("--device-as-default-execution-space"); // When declarations don't mention their execution space, assume they're __device__

    if (not language_standard.empty()) {
        ss.str("");
        ss << "--std=" << language_standard;
        compilation_cmdline_args.push_back(ss.str());
    }
    if (compile_in_debug_mode) {
        compilation_cmdline_args.push_back("--device-debug");
    }
    if (generate_line_info) {
        compilation_cmdline_args.push_back("--generate-line-info");
    }

    // Should we pre-include anything?
    // Also, are the pre-includes searched in the headers provided to nvrtcCreateProgram,
    // or in the include path? Or both?

    for(const auto& def : preprocessor_definitions) {
        compilation_cmdline_args.emplace_back("-D");
        compilation_cmdline_args.emplace_back(def);
    }

    for(const auto& def_pair : preprocessor_definition_pairs) {
        compilation_cmdline_args.emplace_back("-D");
        ss.str("");
        ss << def_pair.first << '=' << def_pair.second;
        compilation_cmdline_args.emplace_back(ss.str());
    }

//    compilation_cmdline_args.emplace_back("--pre-include");
//    //    compilation_cmdline_args.emplace_back("<kernel-runner_preinclude.h>");
//    compilation_cmdline_args.emplace_back("stdint.h");

    for(const auto& path : include_dir_paths) {
        compilation_cmdline_args.emplace_back(std::move("-I"));
        compilation_cmdline_args.emplace_back(path);
    }

    for(const auto& path : preinclude_files) {
        compilation_cmdline_args.emplace_back(std::move("-include"));
        compilation_cmdline_args.emplace_back(path);
    }

    if (spdlog::level_is_at_least(spdlog::level::debug)) {
        ss.str("");
        util::implode(compilation_cmdline_args, ss, ' ');
        spdlog::debug("Kernel compilation generated-command-line arguments: \"{}\"", ss.str());
    }
    return compilation_cmdline_args;
}

compilation_result_t compile_kernel_source_to_ptx(
    CUdevice cuDevice,
    const char *kernel_source_code,
    const char* kernel_source_filename,
    const char* kernel_name_for_later_extraction,
    bool compile_in_debug_mode,
    bool generate_line_info,
    const std::string& language_standard,
    const include_dir_paths_t& include_dir_paths = {},
    const include_dir_paths_t& preinclude_files = {},
    const preprocessor_definitions_t& preprocessor_definitions = {},
        // each string has the form "NAME=value" or "NAME" without an equality sign
    const preprocessor_value_definitions_t& preprocessor_definition_pairs = {}
    // Note: Could potentially take more compilation options here. But then - this
    //  is a super-ugly function, I'd rather delete it, not extend it
    )
{
  // Look away, I (= this function) am hideous!

  auto nvrtc_compilation_options = marshal_compilation_options(
      cuDevice,
      compile_in_debug_mode,
      generate_line_info,
      language_standard,
      include_dir_paths,
      preinclude_files,
      preprocessor_definitions,
      preprocessor_definition_pairs);

  // marshal sources and headers

  nvrtcProgram nvrtc_program;
  auto header_names_and_sources = get_standard_header_substitutes(); // could have really used structured binding here.
  auto& header_names = header_names_and_sources.first;
  auto& header_sources = header_names_and_sources.second;

  // We won't actually pass the program source we've been given as the program source as far as NVRTC is concerned.
  // Why? Because if we do that, NSight Compute won't be able to display it

  // compile

  cuda_api_call(nvrtcCreateProgram,
      &nvrtc_program,
      kernel_source_code,
      kernel_source_filename,
      (int) header_sources.size(),
      header_sources.data(),
      header_names.data());

  spdlog::trace("Program created. Now adding name expression \"{}\"", kernel_name_for_later_extraction);

  cuda_api_call(nvrtcAddNameExpression, nvrtc_program, kernel_name_for_later_extraction);

  std::vector<char*> raw_ptr_compilation_options;
  std::transform(
      nvrtc_compilation_options.begin(),nvrtc_compilation_options.end(),std::back_inserter(raw_ptr_compilation_options),
      [](const std::string& opt) { return const_cast<char*>(opt.c_str()); }
  );

  auto res = nvrtcCompileProgram(nvrtc_program, (int) raw_ptr_compilation_options.size(), raw_ptr_compilation_options.data());

//  if (res != NVRTC_SUCCESS) {
//      spdlog::log(spdlog::level::err, "nvrtcCompileProgram() failed: {}", nvrtcGetErrorString(res));
//  }

  maybe_print_compilation_log(res, nvrtc_program);
      // TODO: Should handle this error in a more visible way

  cuda_api_call_( "nvrtcCompileProgram", [&]() { return res; } );
  size_t ptx_size;
  cuda_api_call(nvrtcGetPTXSize, nvrtc_program, &ptx_size);
  auto ptx_buffer = std::make_unique<char[]>(ptx_size);
  cuda_api_call(nvrtcGetPTX, nvrtc_program, ptx_buffer.get());

  // Not destroying the program here; remember to destroy it later
  //  NVRTC_SAFE_CALL_ALT("nvrtcDestroyProgram", nvrtcDestroyProgram(&prog));
  return { nvrtc_program, ptx_buffer.get() };
}

auto build_cuda_kernel(
//    std::integral_constant<execution_ecosystem_t, execution_ecosystem_t::cuda>,
    cuda::device_t cuda_device, // which level device should this be anyway?
    const char* kernel_source_filename,
    const char* kernel_source,
    const char* kernel_function_name,
    bool debug_mode,
    bool generate_line_info,
    const std::string& language_standard,
    const std::vector<std::string>& include_dir_paths,
    const std::vector<std::string>& preinclude_files,
    const preprocessor_definitions_t& preprocessor_definitions,
    const preprocessor_value_definitions_t& preprocessor_value_definitions
    )
{
    // This is the ugliest and dirtiest function in this file!
    // (Or perhaps I should say, together with compile_kernel_source_to_ptx,
    // it's the ugliest and dirtiest.)

    compilation_result_t compilation_result =
        compile_kernel_source_to_ptx(
            cuda_device.id(),
            kernel_source,
            kernel_source_filename,
            kernel_function_name,
            debug_mode,
            generate_line_info,
            language_standard,
            include_dir_paths,
            preinclude_files,
            preprocessor_definitions,
            preprocessor_value_definitions);
    if (compilation_result.ptx.empty()) {
        throw std::runtime_error("PTX compilation failed");
    }
    spdlog::debug("PTX compilation succeeded, PTX length: {} characters.", compilation_result.ptx.length());
    CUmodule module;
    constexpr const unsigned no_options { 0  };
    cuda_api_call(cuModuleLoadDataEx, &module, compilation_result.ptx.c_str(), no_options, nullptr, nullptr);
    spdlog::debug("PTX loaded as a CUDA module.");
    const char* lowered_name;
    CUfunction kernel_function;
    cuda_api_call(nvrtcGetLoweredName, compilation_result.program, kernel_function_name, &lowered_name);
    spdlog::debug("'Lowered' kernel name is \"" + std::string(lowered_name) + "\"");
    cuda_api_call(cuModuleGetFunction, &kernel_function, module, lowered_name);
    spdlog::debug("Built kernel function pointer obtained from module.");
    cuda_api_call(nvrtcDestroyProgram, &compilation_result.program);
    return std::make_tuple(module, kernel_function, std::move(compilation_result.ptx));
}



#endif  // KERNEL_RUNNER_NVRTC_WRAPPER_HPP_
