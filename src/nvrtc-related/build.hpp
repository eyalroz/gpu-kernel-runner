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
#include "preprocessor_definitions.hpp"

#include "../util/miscellany.hpp"
#include "../util/spdlog-extra.hpp"

#include <cuda/rtc.hpp>
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

using valued_preprocessor_definitions_t = std::unordered_map<std::string, std::string>;
using preprocessor_definitions_t = std::unordered_set<std::string>;

// This is a stub. It's not terribly hard to do, but it's more than a few lines
// of code - certainly if you want it to work on MacOs Windows as well as Linux.
optional<std::string> locate_cuda_include_directory();

struct compilation_result_t {
    bool succeeded;
    optional<std::string> log;
    optional<cuda::module_t> module;
    optional<cuda::kernel_t> kernel;
    optional<std::string> ptx;
};

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
    const std::vector<std::string>& extra_compilation_options);


#endif  // KERNEL_RUNNER_NVRTC_WRAPPER_HPP_
