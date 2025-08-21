#ifndef PARSED_COMMAND_LINE_OPTIONS_HPP_
#define PARSED_COMMAND_LINE_OPTIONS_HPP_

#include "common_types.hpp" // for execution_ecosystem_t
#include "preprocessor_definitions.hpp"
#include "launch_configuration.hpp"

#include "util/filesystem.hpp"
#include "util/optional_and_any.hpp"
#include "util/miscellany.hpp"
#include "util/warning_suppression.hpp"

#include <string>
#include <cstdlib>

enum class early_exit_action_t : int {
    print_help,
    list_kernels,
    list_opencl_platforms
};

// These options are common, and relevant, to any and all kernel adapters
struct parsed_cmdline_options_t {
    bool valid;
    optional<early_exit_action_t> early_exit_action; // as requested by user or resolved based on various arguments
    optional<std::string> help_text;
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
    split_preprocessor_definitions_t preprocessor_definitions;
    std::vector<std::string> extra_compilation_options;

    // std::unordered_map<std::string, std::string> raw_output_size_settings;
    std::unordered_map<std::string, std::size_t> output_buffer_sizes;

    argument_values_t aliased_kernel_arguments;
    include_paths_t include_dir_paths;
    include_paths_t preinclude_files;
    bool zero_output_buffers;
    bool clear_l2_cache;
    bool sync_after_kernel_execution;
    bool sync_after_buffer_op;
    bool write_output_buffers_to_files;
    bool overwrite_allowed;
    bool write_ptx_to_file;
    bool always_print_compilation_log;
    bool write_compilation_log;
    bool print_execution_durations;
    bool generate_source_line_info;
    bool set_default_compilation_options; // target GPU, language standard etc.
    bool compile_only;
    bool generate_debug_info;
    bool accept_oversized_inputs;
    filesystem::path ptx_output_file;
    filesystem::path compilation_log_file;
    filesystem::path execution_durations_file;
    optional<std::string> language_standard; // At the moment, possible (engaged) values are: "c++11","c++14", "c++17"
    bool time_with_events;
    optional_launch_config_components_t forced_launch_config_components;
};

#endif /* PARSED_COMMAND_LINE_OPTIONS_HPP_ */
