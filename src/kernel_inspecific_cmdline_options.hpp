#ifndef KERNEL_INSPECIFIC_COMMAND_LINE_OPTIONS_HPP_
#define KERNEL_INSPECIFIC_COMMAND_LINE_OPTIONS_HPP_

#include "common_types.hpp" // for execution_ecosystem_t

#include "launch_configuration.hpp"

#include <util/filesystem.hpp>
#include <util/optional_and_any.hpp>
#include <util/miscellany.hpp>
#include <util/miscellany.hpp>
#include <util/warning_suppression.hpp>

#include <string>
#include <cstdlib>

// These options are common, and relevant, to any and all kernel adapters
struct kernel_inspecific_cmdline_options_t {
    struct {
        std::string key;
        std::string function_name;
        filesystem::path source_file;
    } kernel;
    execution_ecosystem_t gpu_ecosystem;
    optional<unsigned> platform_id; // Will be nullopt for CUDA
    device_id_t gpu_device_id;
    std::size_t num_runs;
    struct {
        filesystem::path input, output;
    } buffer_base_paths;
    filesystem::path kernel_sources_base_path;
    preprocessor_definitions_t preprocessor_definitions;

    /**
     * Preprocessor definitions specified on the command-line using the
     * defined term as the command-line option, e.g. "--FOO=bar" rather than
     * "-D FOO=bar".
     */
    preprocessor_value_definitions_t preprocessor_value_definitions;
    include_paths_t include_dir_paths;
    include_paths_t preinclude_files;
    bool zero_output_buffers;
    bool write_output_buffers_to_files;
    bool overwrite_allowed;
    bool write_ptx_to_file;
    bool generate_line_info;
    bool compile_only;
//    bool compare_outputs_against_expected;
    bool compile_in_debug_mode;
    filesystem::path ptx_output_file;
    std::string language_standard; // At the moment, possible values are: empty, "c++11","c++14", "c++17"
    bool time_with_events;
    optional_launch_config_components forced_launch_config_components;
};

#endif /* KERNEL_INSPECIFIC_COMMAND_LINE_OPTIONS_HPP_ */
